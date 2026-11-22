#include "parsers.h"

// FunctionGemma tool calls; see the r19 notes in the kepler REFERENCES for the format and why this reuses
// the gemma4 rule names and mapper.

namespace workaround {

// The FunctionGemma template needs a "name" on every role:tool message; OpenAI-style clients send only
// tool_call_id, so fill the name in from the matching earlier assistant tool call (by id, else by order).
static void name_tool_responses_functiongemma(json & messages) {
    json pending = json::array();
    size_t next = 0;
    for (auto & msg : messages) {
        const auto role = msg.value("role", "");
        if (role == "assistant") {
            pending = (msg.contains("tool_calls") && msg.at("tool_calls").is_array()) ? msg.at("tool_calls") : json::array();
            next = 0;
            continue;
        }
        if (role != "tool" || (msg.contains("name") && msg.at("name").is_string())) {
            continue;
        }
        std::string name;
        const auto id = msg.value("tool_call_id", "");
        for (const auto & tc : pending) {
            if (!id.empty() && tc.value("id", "") == id && tc.contains("function")) {
                name = tc.at("function").value("name", "");
                break;
            }
        }
        if (name.empty() && next < pending.size() && pending[next].contains("function")) {
            name = pending[next].at("function").value("name", "");
        }
        next++;
        if (!name.empty()) {
            msg["name"] = name;
        }
    }
}

}

common_chat_params common_chat_params_init_functiongemma(const common_chat_template &      tmpl,
                                                         const autoparser::generation_params & inputs) {
    autoparser::generation_params adjusted = inputs;
    workaround::name_tool_responses_functiongemma(adjusted.messages);

    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, adjusted);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, adjusted);

    data.message_delimiters = {
        { COMMON_CHAT_ROLE_USER,      "<start_of_turn>user"  },
        { COMMON_CHAT_ROLE_ASSISTANT, "<start_of_turn>model" },
    };

    // same argument syntax and AST rule names as gemma4: reuse its mapper
    data.format = COMMON_CHAT_FORMAT_PEG_GEMMA4;

    data.preserved_tokens = {
        "<start_function_call>",
        "<end_function_call>",
        "<start_function_response>",
        "<escape>",
    };
    // after a call the model goes on to the response slot; generation ends there
    data.additional_stops = { "<start_function_response>" };

    auto has_tools       = adjusted.tools.is_array() && !adjusted.tools.empty();
    auto include_grammar = has_tools && adjusted.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start = p.rule("start", p.optional(p.literal("<start_of_turn>model\n")));

        if (include_grammar) {
            // rule names must match gemma4_to_json() in chat-peg-parser.cpp
            p.rule("gemma4-string-content", p.until("<escape>"));
            p.rule("gemma4-string", p.literal("<escape>") + p.ref("gemma4-string-content") + p.literal("<escape>"));
            p.rule("gemma4-bool", p.json_bool());
            p.rule("gemma4-null", p.json_null());
            p.rule("gemma4-number", p.json_number());
            p.rule("gemma4-dict-key", p.rule("gemma4-dict-key-name", p.chars("[^:}]", 1, -1)) + p.literal(":"));
            p.rule("gemma4-dict-kv", p.ref("gemma4-dict-key") + p.space() + p.ref("gemma4-value"));
            p.rule("gemma4-dict", [&]() {
                auto ws      = p.space();
                auto member  = p.ref("gemma4-dict-kv");
                auto members = p.sequence({member, p.zero_or_more(p.sequence({p.literal(","), ws, member}))});
                return p.sequence({
                    p.literal("{"), ws,
                    p.choice({p.literal("}"), p.sequence({members, ws, p.literal("}")})})
                });
            });
            p.rule("gemma4-array", [&]() {
                auto ws       = p.space();
                auto value    = p.ref("gemma4-value");
                auto elements = p.sequence({value, p.zero_or_more(p.sequence({p.literal(","), ws, value}))});
                return p.sequence({
                    p.literal("["), ws,
                    p.choice({p.literal("]"), p.sequence({elements, ws, p.literal("]")})})
                });
            });
            p.rule("gemma4-value", [&]() {
                return p.choice({
                    p.ref("gemma4-string"), p.ref("gemma4-dict"), p.ref("gemma4-array"),
                    p.ref("gemma4-number"), p.ref("gemma4-bool"), p.ref("gemma4-null")
                });
            });

            auto tool_choice = p.choice();
            foreach_function(adjusted.tools, [&](size_t tool_index, const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                tool_choice |= p.rule("tool-" + std::to_string(tool_index), p.tool(p.sequence({
                    p.tool_open(p.tool_name(p.literal(name)) + p.peek(p.literal("{"))),
                    p.tool_args(p.ref("gemma4-dict")),
                })));
            });

            auto tool_call = p.trigger_rule("tool-call", p.repeat(
                "<start_function_call>call:" + tool_choice + "<end_function_call>",
                /* min = */ adjusted.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0,
                /* max = */ adjusted.parallel_tool_calls ? -1 : 1
            ));

            if (adjusted.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return start + tool_call;
            }
            auto content = p.rule("content", p.content(p.until("<start_function_call>")));
            return start + content + tool_call;
        }

        auto content = p.rule("content", p.content(p.until("<end_of_turn>")));
        return start + content;
    });

    if (include_grammar) {
        data.grammar_lazy = adjusted.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });
        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<start_function_call>" },
        };
    }

    return data;
}
