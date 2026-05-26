//! Flow scheduling and storage.
//!
//! The flow data model and on-disk storage live in the
//! [`metalcraft_flows`](https://github.com/rust4ai/metalcraft-flows) crate.
//! This module re-exports those types and keeps two starkbot-specific helpers:
//!
//! - [`walk_flow_prompts`] — extract the `prompt` strings from every reachable
//!   [`FlowNodeType::Core(CoreNodeType::Prompt)`] node in BFS order.
//! - [`append_flow_log`] — a `()`-returning wrapper around the underlying
//!   `io::Result` so existing call sites don't need to change.

pub use metalcraft_flows::{
    delete_flow, list_flows, load_flow, load_flow_logs, save_flow, CoreNodeType, FlowDefinition,
    FlowEdge, FlowLogEntry, FlowNode, FlowNodeType, FlowSummary, SavedFlow,
};

use std::path::Path;

/// Walk a flow graph starting from its `Entry` node and collect the `prompt`
/// text from every reachable `Prompt` node in BFS order.
///
/// Empty / missing `prompt` fields are skipped.
pub fn walk_flow_prompts(flow: &FlowDefinition) -> Vec<String> {
    let mut prompts = Vec::new();
    metalcraft_flows::walk_bfs(flow, |node| {
        if matches!(node.node_type, FlowNodeType::Core(CoreNodeType::Prompt)) {
            if let Some(text) = node.data.get("prompt").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    prompts.push(text.to_string());
                }
            }
        }
    });
    prompts
}

/// Append a log entry, ignoring I/O errors. Wraps
/// [`metalcraft_flows::append_flow_log`] so existing call sites that ignore
/// the return value keep compiling without `let _ =`.
pub fn append_flow_log(log_path: &Path, entry: &FlowLogEntry) {
    let _ = metalcraft_flows::append_flow_log(log_path, entry);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_node(id: &str, node_type: FlowNodeType, data: serde_json::Value) -> FlowNode {
        FlowNode { id: id.to_string(), node_type, data, position: [0.0, 0.0] }
    }

    fn make_edge(id: &str, source: &str, target: &str) -> FlowEdge {
        FlowEdge {
            id: id.to_string(),
            source: source.to_string(),
            target: target.to_string(),
            source_handle: None,
            target_handle: None,
        }
    }

    fn entry() -> FlowNodeType { FlowNodeType::Core(CoreNodeType::Entry) }
    fn prompt() -> FlowNodeType { FlowNodeType::Core(CoreNodeType::Prompt) }
    fn branch() -> FlowNodeType { FlowNodeType::Core(CoreNodeType::Branch) }

    #[test]
    fn walk_flow_prompts_empty_flow() {
        let flow = FlowDefinition { nodes: vec![], edges: vec![] };
        assert!(walk_flow_prompts(&flow).is_empty());
    }

    #[test]
    fn walk_flow_prompts_no_entry_node() {
        let flow = FlowDefinition {
            nodes: vec![make_node("p1", prompt(), json!({"prompt": "hello"}))],
            edges: vec![],
        };
        assert!(walk_flow_prompts(&flow).is_empty());
    }

    #[test]
    fn walk_flow_prompts_entry_only() {
        let flow = FlowDefinition {
            nodes: vec![make_node("entry", entry(), json!({}))],
            edges: vec![],
        };
        assert!(walk_flow_prompts(&flow).is_empty());
    }

    #[test]
    fn walk_flow_prompts_linear_chain() {
        let flow = FlowDefinition {
            nodes: vec![
                make_node("entry", entry(), json!({})),
                make_node("p1", prompt(), json!({"prompt": "first prompt"})),
                make_node("p2", prompt(), json!({"prompt": "second prompt"})),
            ],
            edges: vec![make_edge("e1", "entry", "p1"), make_edge("e2", "p1", "p2")],
        };
        assert_eq!(walk_flow_prompts(&flow), vec!["first prompt", "second prompt"]);
    }

    #[test]
    fn walk_flow_prompts_skips_empty_prompt() {
        let flow = FlowDefinition {
            nodes: vec![
                make_node("entry", entry(), json!({})),
                make_node("p1", prompt(), json!({"prompt": ""})),
                make_node("p2", prompt(), json!({"prompt": "real prompt"})),
            ],
            edges: vec![make_edge("e1", "entry", "p1"), make_edge("e2", "p1", "p2")],
        };
        assert_eq!(walk_flow_prompts(&flow), vec!["real prompt"]);
    }

    #[test]
    fn walk_flow_prompts_skips_branch_nodes() {
        let flow = FlowDefinition {
            nodes: vec![
                make_node("entry", entry(), json!({})),
                make_node("b1", branch(), json!({})),
                make_node("p1", prompt(), json!({"prompt": "after branch"})),
            ],
            edges: vec![make_edge("e1", "entry", "b1"), make_edge("e2", "b1", "p1")],
        };
        assert_eq!(walk_flow_prompts(&flow), vec!["after branch"]);
    }

    #[test]
    fn walk_flow_prompts_disconnected_prompt_ignored() {
        let flow = FlowDefinition {
            nodes: vec![
                make_node("entry", entry(), json!({})),
                make_node("p1", prompt(), json!({"prompt": "connected"})),
                make_node("p2", prompt(), json!({"prompt": "disconnected"})),
            ],
            edges: vec![make_edge("e1", "entry", "p1")],
        };
        assert_eq!(walk_flow_prompts(&flow), vec!["connected"]);
    }

    #[test]
    fn walk_flow_prompts_no_cycle_loop() {
        let flow = FlowDefinition {
            nodes: vec![
                make_node("entry", entry(), json!({})),
                make_node("p1", prompt(), json!({"prompt": "one"})),
                make_node("p2", prompt(), json!({"prompt": "two"})),
            ],
            edges: vec![
                make_edge("e1", "entry", "p1"),
                make_edge("e2", "p1", "p2"),
                make_edge("e3", "p2", "p1"),
            ],
        };
        assert_eq!(walk_flow_prompts(&flow), vec!["one", "two"]);
    }

    #[test]
    fn walk_flow_prompts_missing_prompt_field() {
        let flow = FlowDefinition {
            nodes: vec![
                make_node("entry", entry(), json!({})),
                make_node("p1", prompt(), json!({"label": "no prompt field"})),
            ],
            edges: vec![make_edge("e1", "entry", "p1")],
        };
        assert!(walk_flow_prompts(&flow).is_empty());
    }
}
