from workflow.runner import run_message_flow, run_career_conversation


def main() -> None:
    run_message_flow(enable_memory=True)


if __name__ == "__main__":
    main()