from adapter import on_message


def test_context_guard_on_prior_history():
    history = [{"role": "user", "content": "bonjour"}]
    decision = on_message(history, "go")
    assert decision.action == "ask_new_conversation"


def test_clean_context_allows_run():
    decision = on_message([], "go")
    assert decision.action == "run_bench"


if __name__ == "__main__":
    test_context_guard_on_prior_history()
    test_clean_context_allows_run()
    print("context guard ok")
