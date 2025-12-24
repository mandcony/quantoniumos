#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
CLI Chat Interface for QuantoniumOS
Uses the configured local LLM (Llama 3) via ai_model_wrapper.
"""
from __future__ import annotations

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.apps.ai_model_wrapper import format_prompt_auto, generate_response, get_configured_model_id

def main():
    print("QuantoniumOS CLI Chat")
    print("---------------------")
    print(f"Model: {get_configured_model_id()}")
    print("Type 'exit' or 'quit' to stop.\n")

    history = []

    # Non-interactive mode (piped input): read once, respond once, exit.
    if not sys.stdin.isatty():
        user_text = sys.stdin.read().strip()
        if not user_text:
            return
        full_prompt = format_prompt_auto(
            user_text,
            history=None,
            system_prompt="You are a helpful assistant running locally on QuantoniumOS.",
        )
        response = generate_response(full_prompt, max_tokens=200)
        print(response.strip())
        return

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ('exit', 'quit'):
                break
            
            if not user_input.strip():
                continue

            print("Assistant: ", end="", flush=True)
            
            # Simple prompt formatting for the wrapper
            # The wrapper handles history formatting if we passed it, 
            # but generate_response takes a raw string prompt.
            # We need to format it ourselves or update the wrapper usage.
            
            # Let's use the wrapper's format_chat_prompt if available, 
            # but I'll just construct it here to be safe and simple for the CLI.
            full_prompt = format_prompt_auto(
                user_input,
                history=history,
                system_prompt="You are a helpful assistant running locally on QuantoniumOS."
            )
            
            response = generate_response(full_prompt, max_tokens=200)
            
            print(response.strip())
            print()
            
            history.append((user_input, response))
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
