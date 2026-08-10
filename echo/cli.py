"""A minimal terminal loop for exercising the library. Not a product UI."""

from __future__ import annotations

import argparse
import sys

from . import storage
from .agent import Agent
from .conversation import Conversation
from .extraction import LLMMemoryExtractor
from .llm import DEFAULT_MODEL, AnthropicLLM, LLMError

HELP = """\
commands:
  /remember   run memory extraction over this conversation
  /recall X   search persistent memory for X and load hits into this turn
  /memories   list everything in persistent memory
  /context    show the temporary context (discarded when you quit)
  /forget     clear the temporary context
  /quit       leave
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="echo", description="Talk to ECHO.")
    parser.add_argument("--resume", metavar="ID", help="continue a saved conversation")
    parser.add_argument("--list", action="store_true", help="list saved conversation ids")
    parser.add_argument("--system", help="system prompt for a new conversation")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dir", default=str(storage.DEFAULT_ROOT), help="data root directory")
    return parser


def _handle_command(agent: Agent, line: str) -> bool:
    """Run a slash command. Returns False if the loop should exit."""
    command, _, argument = line.partition(" ")
    argument = argument.strip()

    if command in ("/quit", "/exit"):
        return False

    if command == "/help":
        print(HELP)
    elif command == "/remember":
        created = agent.remember()
        if created:
            for memory in created:
                print(f"  + [{memory.memory_type.value}] {memory.content}")
        else:
            print("  (nothing worth remembering)")
    elif command == "/recall":
        if not argument:
            print("  usage: /recall <text>")
        else:
            found = agent.recall(argument)
            for memory in found:
                print(f"  ~ [{memory.memory_type.value}] {memory.content}")
            if not found:
                print("  (no matching memories)")
    elif command == "/memories":
        everything = agent.memories.all()
        for memory in everything:
            print(
                f"  [{memory.memory_type.value}] {memory.content} "
                f"(conf {memory.confidence:.2f}, imp {memory.importance:.2f}, "
                f"accessed {memory.access_count}x)"
            )
        if not everything:
            print("  (persistent memory is empty)")
    elif command == "/context":
        print("  " + (agent.context.render() or "(temporary context is empty)"))
    elif command == "/forget":
        agent.context.clear()
        print("  temporary context cleared")
    else:
        print(f"  unknown command {command!r} — try /help")
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        ids = storage.list_ids(storage.conversations_dir(args.dir))
        print("\n".join(ids) if ids else "(no saved conversations)")
        return 0

    llm = AnthropicLLM(model=args.model)
    extractor = LLMMemoryExtractor(llm)

    if args.resume:
        try:
            agent = Agent.resume(args.resume, llm, directory=args.dir, extractor=extractor)
        except FileNotFoundError:
            print(f"no saved conversation with id {args.resume!r}", file=sys.stderr)
            return 1
        print(f"resumed {agent.id} ({len(agent.conversation)} messages)")
    else:
        agent = Agent(
            llm,
            Conversation(system=args.system),
            directory=args.dir,
            extractor=extractor,
        )
        print(f"new conversation {agent.id}")

    print(f"{len(agent.memories)} memories on file. /help for commands.\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue

        if text.startswith("/"):
            if not _handle_command(agent, text):
                break
            continue

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
