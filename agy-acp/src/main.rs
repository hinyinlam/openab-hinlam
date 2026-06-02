use fs2::FileExt;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use tokio::process::Command;
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    id: Option<u64>,
    method: Option<String>,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcNotification {
    jsonrpc: &'static str,
    method: String,
    params: Value,
}

/// Persisted session→conversation mapping stored in ~/.openab/agy-acp/sessions.json
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SessionStore {
    sessions: HashMap<String, StoredSession>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredSession {
    conversation_id: Option<String>,
    /// Last step idx read from SQLite; used for delta extraction.
    #[serde(default)]
    last_step_idx: i64,
}

struct Session {
    conversation_id: Option<String>,
    /// Last step idx read from SQLite.
    last_step_idx: i64,
}

struct Adapter {
    sessions: HashMap<String, Session>,
    working_dir: String,
    conversations_dir: PathBuf,
    state_file: PathBuf,
}

impl Adapter {
    fn new() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let state_dir = PathBuf::from(&home).join(".openab/agy-acp");
        Self {
            sessions: HashMap::new(),
            working_dir: std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| "/tmp".to_string()),
            conversations_dir: PathBuf::from(&home).join(".gemini/antigravity-cli/conversations"),
            state_file: state_dir.join("sessions.json"),
        }
    }

    /// Acquire exclusive lock on a dedicated lock file for read-write mutual exclusion.
    fn lock_state_file(&self) -> Option<fs::File> {
        if let Some(parent) = self.state_file.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let lock_path = self.state_file.with_extension("lock");
        let lock_file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .ok()?;
        lock_file.lock_exclusive().ok()?;
        Some(lock_file)
    }

    /// Load persisted session store (caller must hold lock).
    fn load_store_inner(&self) -> SessionStore {
        let Some(file) = fs::File::open(&self.state_file).ok() else {
            return SessionStore::default();
        };
        serde_json::from_reader(&file).unwrap_or_default()
    }

    /// Load persisted session store with lock.
    fn load_store(&self) -> SessionStore {
        let _lock = self.lock_state_file();
        self.load_store_inner()
    }

    /// Try to restore conversation_id and last_step_idx from persisted state.
    fn restore_session(&self, session_id: &str) -> Option<(String, i64)> {
        let store = self.load_store();
        store.sessions.get(session_id).and_then(|s| {
            s.conversation_id.clone().map(|cid| (cid, s.last_step_idx))
        })
    }

    /// Persist a session binding (read-modify-write under single lock).
    fn persist_session(&self, session_id: &str, conversation_id: Option<&str>, last_step_idx: i64) {
        let Some(_lock) = self.lock_state_file() else {
            return;
        };
        let mut store = self.load_store_inner();
        store.sessions.insert(
            session_id.to_string(),
            StoredSession {
                conversation_id: conversation_id.map(String::from),
                last_step_idx,
            },
        );
        let tmp = self.state_file.with_extension("tmp");
        if let Ok(file) = fs::File::create(&tmp) {
            if serde_json::to_writer_pretty(&file, &store).is_ok() {
                let _ = fs::rename(&tmp, &self.state_file);
            }
        }
    }

    fn conversation_snapshot(&self) -> HashSet<String> {
        let Ok(entries) = fs::read_dir(&self.conversations_dir) else {
            return HashSet::new();
        };
        entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let path = e.path();
                if path.extension().map(|x| x == "db").unwrap_or(false) {
                    path.file_stem().map(|s| s.to_string_lossy().to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    fn new_conversation_id(&self, before: &HashSet<String>) -> Option<String> {
        let after = self.conversation_snapshot();
        let mut created: Vec<_> = after.difference(before).collect();
        if created.is_empty() {
            return None;
        }
        if created.len() > 1 {
            eprintln!(
                "[agy-acp] WARN: multiple new agy conversation files appeared; \
                 refusing to bind"
            );
            return None;
        }
        Some(created.remove(0).clone())
    }

    /// Extract text from a protobuf blob by reading field 8 (tag 0x42 = field 8, wire type 2).
    fn extract_text_from_step_payload(blob: &[u8]) -> Option<String> {
        let mut i = 0;
        while i < blob.len() {
            let (tag, consumed) = Self::read_varint(&blob[i..])?;
            i += consumed;
            let field_number = tag >> 3;
            let wire_type = tag & 0x7;
            match wire_type {
                0 => {
                    let (_, c) = Self::read_varint(&blob[i..])?;
                    i += c;
                }
                2 => {
                    let (len, c) = Self::read_varint(&blob[i..])?;
                    i += c;
                    let len = len as usize;
                    if i + len > blob.len() {
                        return None;
                    }
                    if field_number == 8 {
                        return String::from_utf8(blob[i..i + len].to_vec()).ok();
                    }
                    i += len;
                }
                5 => { i += 4; }
                1 => { i += 8; }
                _ => return None,
            }
        }
        None
    }

    /// Read a protobuf varint, returning (value, bytes_consumed).
    fn read_varint(buf: &[u8]) -> Option<(u64, usize)> {
        let mut result: u64 = 0;
        let mut shift = 0;
        for (i, &byte) in buf.iter().enumerate() {
            if shift >= 70 {
                return None;
            }
            result |= ((byte & 0x7F) as u64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                return Some((result, i + 1));
            }
        }
        None
    }

    /// Read the latest response from the SQLite conversation DB.
    /// Returns (response_text, max_step_idx) or None if reading fails.
    fn read_response_from_db(&self, conversation_id: &str, after_step_idx: i64) -> Option<(String, i64)> {
        let db_path = self.conversations_dir.join(format!("{}.db", conversation_id));
        let conn = Connection::open_with_flags(
            &db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        ).ok()?;
        let mut stmt = conn.prepare(
            "SELECT idx, step_payload FROM steps WHERE idx > ?1 ORDER BY idx"
        ).ok()?;
        let rows: Vec<(i64, Vec<u8>)> = stmt.query_map([after_step_idx], |row| {
            Ok((row.get(0)?, row.get(1)?))
        }).ok()?.filter_map(|r| r.ok()).collect();

        let mut max_idx = after_step_idx;
        let mut response_parts: Vec<String> = Vec::new();
        for (idx, payload) in &rows {
            max_idx = max_idx.max(*idx);
            if let Some(text) = Self::extract_text_from_step_payload(payload) {
                if !text.is_empty() {
                    response_parts.push(text);
                }
            }
        }
        if response_parts.is_empty() {
            return None;
        }
        Some((response_parts.join("\n"), max_idx))
    }

    fn evict_if_needed(&mut self) {
        const MAX_SESSIONS: usize = 64;
        while self.sessions.len() >= MAX_SESSIONS {
            if let Some(key) = self.sessions.keys().next().cloned() {
                self.sessions.remove(&key);
            }
        }
    }

    fn restore_session_state(&mut self, session_id: &str) -> bool {
        let Some((conversation_id, last_step_idx)) = self.restore_session(session_id) else {
            return false;
        };
        if !self.sessions.contains_key(session_id) {
            self.evict_if_needed();
        }
        self.sessions.insert(
            session_id.to_string(),
            Session {
                conversation_id: Some(conversation_id),
                last_step_idx,
            },
        );
        true
    }

    fn handle_initialize(&self, id: u64) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": 1,
                "agentInfo": { "name": "agy", "version": env!("CARGO_PKG_VERSION") },
                "agentCapabilities": { "streaming": true, "loadSession": true },
            })),
            error: None,
        }
    }

    fn handle_session_new(&mut self, id: u64) -> JsonRpcResponse {
        let session_id = Uuid::new_v4().to_string();
        self.evict_if_needed();
        self.sessions.insert(
            session_id.clone(),
            Session {
                conversation_id: None,
                last_step_idx: -1,
            },
        );
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "sessionId": session_id })),
            error: None,
        }
    }

    fn handle_session_load(&mut self, id: u64, params: &Value) -> JsonRpcResponse {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if session_id.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId"})),
            };
        }

        if self.restore_session_state(session_id) {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({ "sessionId": session_id })),
                error: None,
            };
        }

        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(json!({
                "code": -32000,
                "message": format!("unknown sessionId: {session_id}"),
            })),
        }
    }

    async fn handle_session_prompt(&mut self, id: u64, params: &Value) -> Vec<String> {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Restore evicted session from state file if needed
        if !session_id.is_empty() && !self.sessions.contains_key(session_id) {
            let _ = self.restore_session_state(session_id);
        }

        let prompt_text = params
            .get("prompt")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        let clean_prompt = prompt_text.trim();

        // Take snapshot before spawning agy if we need to bind a conversation
        let snapshot = if self
            .sessions
            .get(session_id)
            .map(|s| s.conversation_id.is_none())
            .unwrap_or(false)
        {
            Some(self.conversation_snapshot())
        } else {
            None
        };

        // Build args
        let mut args: Vec<String> = Vec::new();
        args.push("--add-dir".to_string());
        args.push(self.working_dir.clone());
        if let Ok(extra) = std::env::var("AGY_EXTRA_ARGS") {
            args.extend(extra.split_whitespace().map(String::from));
        }
        if let Some(session) = self.sessions.get(session_id) {
            if let Some(conv_id) = &session.conversation_id {
                args.push("--conversation".to_string());
                args.push(conv_id.clone());
            }
        }
        args.push("-p".to_string());
        args.push(clean_prompt.to_string());

        let result = Command::new("agy")
            .args(&args)
            .current_dir(&self.working_dir)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .await;

        let mut output_lines = Vec::new();

        match result {
            Ok(output) => {
                let stderr_text = String::from_utf8_lossy(&output.stderr);
                if !stderr_text.is_empty() {
                    eprintln!("[agy-acp] agy stderr: {}", stderr_text.trim_end());
                }

                if !output.status.success() {
                    eprintln!("[agy-acp] WARN: agy exited with status: {}", output.status);
                    if output.stdout.is_empty() {
                        let msg = if stderr_text.is_empty() {
                            format!("agy exited with status: {}", output.status)
                        } else {
                            format!("agy failed: {}", stderr_text.trim_end())
                        };
                        let resp = JsonRpcResponse {
                            jsonrpc: "2.0",
                            id,
                            result: None,
                            error: Some(json!({"code":-32000,"message":msg})),
                        };
                        output_lines.push(serde_json::to_string(&resp).unwrap());
                        return output_lines;
                    }
                }

                let full_text = String::from_utf8_lossy(&output.stdout).to_string();

                // Bind conversation from snapshot diff
                let conv_id = snapshot
                    .as_ref()
                    .and_then(|before| self.new_conversation_id(before));

                if let Some(session) = self.sessions.get_mut(session_id) {
                    if session.conversation_id.is_none() {
                        session.conversation_id = conv_id.clone();
                    }
                }

                let bound_conv_id = self.sessions.get(session_id).and_then(|s| s.conversation_id.clone());
                let last_step_idx = self.sessions.get(session_id).map(|s| s.last_step_idx).unwrap_or(-1);

                // Read response delta from SQLite
                let (new_text, new_step_idx) = if let Some(cid) = &bound_conv_id {
                    match self.read_response_from_db(cid, last_step_idx) {
                        Some((text, idx)) => {
                            eprintln!("[agy-acp] delta from SQLite (steps {} → {})", last_step_idx, idx);
                            (text, idx)
                        }
                        None => {
                            // DB read failed — use full stdout as last resort
                            eprintln!("[agy-acp] WARN: SQLite read returned no text, using stdout");
                            (full_text.clone(), last_step_idx)
                        }
                    }
                } else {
                    eprintln!("[agy-acp] WARN: could not bind conversation ID; single-turn mode");
                    (full_text.clone(), -1i64)
                };

                // Persist session state
                if let Some(session) = self.sessions.get_mut(session_id) {
                    if session.conversation_id.is_some() {
                        session.last_step_idx = new_step_idx;
                    }
                }
                if bound_conv_id.is_some() {
                    self.persist_session(session_id, bound_conv_id.as_deref(), new_step_idx);
                }

                let notification = serde_json::to_string(&JsonRpcNotification {
                    jsonrpc: "2.0",
                    method: "session/update".to_string(),
                    params: json!({
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": { "type": "text", "text": new_text },
                        },
                    }),
                })
                .unwrap();
                output_lines.push(notification);
                let resp = JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({ "stopReason": "end_turn" })),
                    error: None,
                };
                output_lines.push(serde_json::to_string(&resp).unwrap());
            }
            Err(e) => {
                let resp = JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: None,
                    error: Some(json!({"code":-32000,"message":format!("failed to run agy: {e}")})),
                };
                output_lines.push(serde_json::to_string(&resp).unwrap());
            }
        }
        output_lines
    }
}

#[tokio::main]
async fn main() {
    let mut adapter = Adapter::new();

    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) if !l.trim().is_empty() => {
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
                _ => {}
            }
        }
    });

    let mut stdout = io::stdout();

    while let Some(line) = rx.recv().await {
        let req: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let id = match req.id {
            Some(id) => id,
            None => continue,
        };

        let output = match req.method.as_deref() {
            Some("initialize") => {
                vec![serde_json::to_string(&adapter.handle_initialize(id)).unwrap()]
            }
            Some("session/new") => {
                vec![serde_json::to_string(&adapter.handle_session_new(id)).unwrap()]
            }
            Some("session/load") => {
                let params = req.params.unwrap_or(json!({}));
                vec![serde_json::to_string(&adapter.handle_session_load(id, &params)).unwrap()]
            }
            Some("session/prompt") => {
                let params = req.params.unwrap_or(json!({}));
                adapter.handle_session_prompt(id, &params).await
            }
            Some("session/cancel") => {
                let r = JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({})),
                    error: None,
                };
                vec![serde_json::to_string(&r).unwrap()]
            }
            Some(method) => {
                let r = JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: None,
                    error: Some(
                        json!({"code":-32601,"message":format!("method not found: {method}")}),
                    ),
                };
                vec![serde_json::to_string(&r).unwrap()]
            }
            None => continue,
        };

        for line in output {
            let _ = writeln!(stdout, "{}", line);
        }
        let _ = stdout.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_from_step_payload_field_8() {
        // Simulate a protobuf with field 8 (tag 0x42) containing "hello"
        let mut blob = Vec::new();
        // field 1, varint, value 3
        blob.push(0x08); blob.push(0x03);
        // field 8, length-delimited, "hello"
        blob.push(0x42); blob.push(0x05);
        blob.extend_from_slice(b"hello");
        assert_eq!(Adapter::extract_text_from_step_payload(&blob), Some("hello".to_string()));
    }

    #[test]
    fn test_extract_text_skips_non_field_8() {
        // Only field 1 (varint) — no field 8
        let blob = vec![0x08, 0x03];
        assert_eq!(Adapter::extract_text_from_step_payload(&blob), None);
    }

    #[test]
    fn test_extract_text_multiline() {
        let text = b"Safe memory rules\nCompiler points out the flaws\nFast and fearless code";
        let mut blob = Vec::new();
        blob.push(0x08); blob.push(0x01); // field 1 varint
        blob.push(0x42); // field 8, LEN
        // length as varint (70 bytes)
        blob.push(text.len() as u8);
        blob.extend_from_slice(text);
        assert_eq!(
            Adapter::extract_text_from_step_payload(&blob),
            Some("Safe memory rules\nCompiler points out the flaws\nFast and fearless code".to_string())
        );
    }

    #[test]
    fn test_read_varint() {
        assert_eq!(Adapter::read_varint(&[0x05]), Some((5, 1)));
        assert_eq!(Adapter::read_varint(&[0xAC, 0x02]), Some((300, 2)));
        assert_eq!(Adapter::read_varint(&[]), None);
    }

    #[test]
    fn test_initialize_advertises_load_session_support() {
        let adapter = Adapter::new();
        let response = adapter.handle_initialize(1);
        assert_eq!(
            response
                .result
                .as_ref()
                .and_then(|r| r.get("agentCapabilities"))
                .and_then(|c| c.get("loadSession"))
                .and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_session_load_restores_persisted_session() {
        let root = std::env::temp_dir().join(format!("agy-acp-load-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
        };
        adapter.persist_session("sess-1", Some("conv-abc"), 5);

        let response = adapter.handle_session_load(7, &json!({"sessionId": "sess-1"}));
        assert!(response.error.is_none());
        assert_eq!(
            adapter.sessions.get("sess-1").and_then(|s| s.conversation_id.as_deref()),
            Some("conv-abc")
        );
        assert_eq!(
            adapter.sessions.get("sess-1").map(|s| s.last_step_idx),
            Some(5)
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_session_load_rejects_unknown_session() {
        let root = std::env::temp_dir().join(format!("agy-acp-missing-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
        };

        let response = adapter.handle_session_load(9, &json!({"sessionId": "missing"}));
        assert!(response.result.is_none());
        assert_eq!(
            response.error.as_ref().and_then(|e| e.get("message")).and_then(|m| m.as_str()),
            Some("unknown sessionId: missing")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_snapshot_detects_db_conversations() {
        let root = std::env::temp_dir().join(format!("agy-acp-db-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();
        fs::write(conv_dir.join("existing.db"), b"old").unwrap();

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir.clone(),
            state_file: root.join("sessions.json"),
        };

        let before = adapter.conversation_snapshot();
        assert!(before.contains("existing"));

        fs::write(conv_dir.join("new-conv.db"), b"new").unwrap();
        // WAL sidecar files should not be picked up
        fs::write(conv_dir.join("new-conv.db-wal"), b"wal").unwrap();
        fs::write(conv_dir.join("new-conv.db-shm"), b"shm").unwrap();

        assert_eq!(
            adapter.new_conversation_id(&before),
            Some("new-conv".to_string())
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_snapshot_ignores_multiple_new_files() {
        let root = std::env::temp_dir().join(format!("agy-acp-multi-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir.clone(),
            state_file: root.join("sessions.json"),
        };

        let before = adapter.conversation_snapshot();
        fs::write(conv_dir.join("a.db"), b"").unwrap();
        fs::write(conv_dir.join("b.db"), b"").unwrap();

        assert_eq!(adapter.new_conversation_id(&before), None);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_persist_and_restore_session() {
        let root = std::env::temp_dir().join(format!("agy-acp-state-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
        };

        adapter.persist_session("sess-1", Some("conv-abc"), 7);
        let restored = adapter.restore_session("sess-1");
        assert_eq!(restored, Some(("conv-abc".to_string(), 7)));

        let missing = adapter.restore_session("sess-unknown");
        assert_eq!(missing, None);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O — requires real SQLite DB
    fn test_read_response_from_db() {
        let root = std::env::temp_dir().join(format!("agy-acp-sqlite-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();

        // Create a test SQLite DB with steps table
        let db_path = conv_dir.join("test-conv.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE steps (
                idx INTEGER PRIMARY KEY,
                step_type INTEGER NOT NULL DEFAULT 0,
                status INTEGER NOT NULL DEFAULT 0,
                has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
                metadata BLOB,
                error_details BLOB,
                permissions BLOB,
                task_details BLOB,
                render_info BLOB,
                step_payload BLOB,
                step_format INTEGER NOT NULL DEFAULT 0
            )"
        ).unwrap();

        // Insert a step with field 8 containing "hello world"
        let mut payload = Vec::new();
        payload.push(0x08); payload.push(0x01); // field 1 varint
        payload.push(0x42); payload.push(11);   // field 8, len=11
        payload.extend_from_slice(b"hello world");

        conn.execute(
            "INSERT INTO steps (idx, step_payload) VALUES (?1, ?2)",
            rusqlite::params![1i64, payload],
        ).unwrap();
        drop(conn);

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir,
            state_file: root.join("sessions.json"),
        };

        let result = adapter.read_response_from_db("test-conv", -1);
        assert_eq!(result, Some(("hello world".to_string(), 1)));

        // Reading after idx 1 should return None (no new steps)
        let result = adapter.read_response_from_db("test-conv", 1);
        assert_eq!(result, None);

        let _ = fs::remove_dir_all(root);
    }
}
