use serde_json::Value;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAX_SEARCH_DEPTH: usize = 8;
const MAX_LINE_CHARS: usize = 500;

#[derive(Debug, Clone, Default)]
pub struct ClaudeActivityTrace {
    enabled: bool,
    session_id: String,
    transcript_path: Option<PathBuf>,
    offset: u64,
}

impl ClaudeActivityTrace {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn new(session_id: Option<&str>) -> Self {
        let Some(session_id) = session_id else {
            return Self::disabled();
        };
        if session_id.is_empty() {
            return Self::disabled();
        }
        Self {
            enabled: true,
            session_id: session_id.to_string(),
            transcript_path: None,
            offset: 0,
        }
    }

    pub fn poll_summary(&mut self) -> Option<String> {
        if !self.enabled {
            return None;
        }
        let path = match self.transcript_path.clone() {
            Some(path) => path,
            None => {
                let path = find_claude_transcript(&self.session_id)?;
                self.offset = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                self.transcript_path = Some(path.clone());
                path
            }
        };
        let (new_offset, events) = read_new_events(&path, self.offset).ok()?;
        self.offset = new_offset;
        if events.is_empty() {
            None
        } else {
            Some(format_activity_summary(&events))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaudeLogEvent {
    AssistantText(String),
    ToolUse { name: String, detail: String },
    ToolResult(String),
}

pub fn find_claude_transcript(session_id: &str) -> Option<PathBuf> {
    let root = claude_projects_dir();
    let filename = format!("{session_id}.jsonl");
    find_named_file(&root, &filename, 0)
}

fn claude_projects_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("CLAUDE_CONFIG_DIR") {
        return PathBuf::from(dir).join("projects");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".claude").join("projects")
}

fn find_named_file(dir: &Path, filename: &str, depth: usize) -> Option<PathBuf> {
    if depth > MAX_SEARCH_DEPTH {
        return None;
    }
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.file_name().and_then(|s| s.to_str()) == Some(filename) {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_named_file(&path, filename, depth + 1) {
                return Some(found);
            }
        }
    }
    None
}

fn read_new_events(path: &Path, offset: u64) -> std::io::Result<(u64, Vec<ClaudeLogEvent>)> {
    let mut file = fs::File::open(path)?;
    let len = file.metadata()?.len();
    let offset = offset.min(len);
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;
    let new_offset = len;
    let events = parse_jsonl_events(&buf);
    Ok((new_offset, events))
}

pub fn parse_jsonl_events(jsonl: &str) -> Vec<ClaudeLogEvent> {
    jsonl
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .flat_map(|value| parse_value_events(&value))
        .collect()
}

fn parse_value_events(value: &Value) -> Vec<ClaudeLogEvent> {
    let mut events = Vec::new();
    let content = value
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array());
    if let Some(content) = content {
        for item in content {
            let Some(kind) = item.get("type").and_then(|v| v.as_str()) else {
                continue;
            };
            match kind {
                "text" => {
                    if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                        if should_emit_text(text) {
                            events.push(ClaudeLogEvent::AssistantText(safe_excerpt(text)));
                        }
                    }
                }
                "tool_use" => {
                    let name = item
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("tool")
                        .to_string();
                    let detail = item.get("input").map(tool_detail).unwrap_or_default();
                    events.push(ClaudeLogEvent::ToolUse {
                        name,
                        detail: safe_excerpt(&detail),
                    });
                }
                "tool_result" => {
                    if let Some(result) = item.get("content").and_then(|v| v.as_str()) {
                        if !result.trim().is_empty() {
                            events.push(ClaudeLogEvent::ToolResult(safe_excerpt(result)));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    if let Some(result) = value.get("toolUseResult") {
        if let Some(stdout) = result.get("stdout").and_then(|v| v.as_str()) {
            if !stdout.trim().is_empty() {
                events.push(ClaudeLogEvent::ToolResult(safe_excerpt(stdout)));
            }
        }
        if let Some(stderr) = result.get("stderr").and_then(|v| v.as_str()) {
            if !stderr.trim().is_empty() {
                events.push(ClaudeLogEvent::ToolResult(safe_excerpt(stderr)));
            }
        }
    }
    events
}

fn should_emit_text(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty()
        && !trimmed.starts_with("<sender_context>")
        && !trimmed.starts_with("<EXTREMELY_IMPORTANT>")
        && !trimmed.contains("signature\":")
}

fn tool_detail(input: &Value) -> String {
    input
        .get("command")
        .or_else(|| input.get("file_path"))
        .or_else(|| input.get("pattern"))
        .and_then(|v| v.as_str())
        .map(ToString::to_string)
        .unwrap_or_else(|| input.to_string())
}

fn safe_excerpt(text: &str) -> String {
    let mut out = redact_secrets(text).replace('\n', "\\n");
    if out.chars().count() > MAX_LINE_CHARS {
        out = out.chars().take(MAX_LINE_CHARS).collect::<String>();
        out.push_str(" …");
    }
    out
}

fn redact_secrets(text: &str) -> String {
    let mut out = String::new();
    for token in text.split_whitespace() {
        if looks_secret(token) {
            out.push_str("[REDACTED]");
        } else {
            out.push_str(token);
        }
        out.push(' ');
    }
    out.trim_end().to_string()
}

fn looks_secret(token: &str) -> bool {
    let trimmed = token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-');
    trimmed.starts_with("ghp_")
        || trimmed.starts_with("ghs_")
        || trimmed.starts_with("github_pat_")
        || (trimmed.len() > 45
            && trimmed
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'))
}

fn format_activity_summary(events: &[ClaudeLogEvent]) -> String {
    let mut lines = vec!["🧵 **Claude Code activity trace**".to_string()];
    for event in events
        .iter()
        .rev()
        .take(8)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        match event {
            ClaudeLogEvent::AssistantText(text) => lines.push(format!("💬 {text}")),
            ClaudeLogEvent::ToolUse { name, detail } => lines.push(format!("🔧 {name}: {detail}")),
            ClaudeLogEvent::ToolResult(result) => lines.push(format!("📄 {result}")),
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_safe_text_tool_and_result_events() {
        let jsonl = r#"
{"message":{"content":[{"type":"text","text":"Working on TQQQ state"}]}}
{"message":{"content":[{"type":"tool_use","name":"Bash","input":{"command":"git status --short"}}]}}
{"message":{"content":[{"type":"tool_result","content":" M src/main.rs"}]}}
{"toolUseResult":{"stdout":"done"}}
"#;
        let events = parse_jsonl_events(jsonl);
        assert_eq!(
            events,
            vec![
                ClaudeLogEvent::AssistantText("Working on TQQQ state".into()),
                ClaudeLogEvent::ToolUse {
                    name: "Bash".into(),
                    detail: "git status --short".into(),
                },
                ClaudeLogEvent::ToolResult("M src/main.rs".into()),
                ClaudeLogEvent::ToolResult("done".into()),
            ]
        );
    }

    #[test]
    fn suppresses_hidden_context_and_redacts_tokens() {
        let jsonl = r#"
{"message":{"content":[{"type":"text","text":"<EXTREMELY_IMPORTANT>secret system prompt</EXTREMELY_IMPORTANT>"}]}}
{"message":{"content":[{"type":"text","text":"token ghp_abcdefghijklmnopqrstuvwxyz1234567890ABCDEFG hidden"}]}}
"#;
        let events = parse_jsonl_events(jsonl);
        assert_eq!(
            events,
            vec![ClaudeLogEvent::AssistantText(
                "token [REDACTED] hidden".into()
            )]
        );
    }
}
