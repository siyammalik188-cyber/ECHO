"""A minimal terminal loop for exercising the library. Not a product UI."""

from __future__ import annotations

import argparse
import sys

from . import storage
from .agent import Agent
from .conversation import Conversation
from .llm import DEFAULT_MODEL, AnthropicLLM, LLMError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="echo", description="Talk to ECHO.")
    parser.add_argument("--resume", metavar="ID", help="continue a saved conversation")
    parser.add_argument("--list", action="store_true", help="list saved conversation ids")
    parser.add_argument("--system", help="system prompt for a new conversation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dir", default=str(storage.DEFAULT_DIR), help="storage directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        ids = storage.list_ids(args.dir)
        print("\n".join(ids) if ids else "(no saved conversations)")
        return 0

    llm = AnthropicLLM(model=args.model)

    if args.resume:
        try:
            agent = Agent.resume(args.resume, llm, directory=args.dir)
        except FileNotFoundError:
            print(f"no saved conversation with id {args.resume!r}", file=sys.stderr)
            return 1
        print(f"resumed {agent.id} ({len(agent.conversation)} messages)")
    else:
        agent = Agent(llm, Conversation(system=args.system), directory=args.dir)
        print(f"new conversation {agent.id}")

    print("type a message, or /quit to leave. every turn is saved.\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break

        try:
            reply = agent.send(text)
        except LLMError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            continue
        print(f"\necho> {reply}\n")
        agent.save()

    path = agent.save()
    print(f"saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
