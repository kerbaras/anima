"""CLI entry point — Rich REPL for the Freudian Mind."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel

from .config import MindConfig
from .mind import FreudianMind

console = Console()


async def run():
    config = MindConfig()
    mind = FreudianMind(config)
    await mind.start()

    conv_id = await mind.new_conversation()

    console.print(
        Panel(
            "[bold]The Freudian Mind[/bold]\n"
            "Unconscious (Opus) | Preconscious (Sonnet) | Conscious (Haiku)\n\n"
            "Commands: [dim]quit, /new, /state[/dim]",
            border_style="blue",
        )
    )

    try:
        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: console.input("[bold green]You:[/bold green] ").strip()
            )

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "/new":
                conv_id = await mind.new_conversation()
                console.print(f"[dim]New conversation: {conv_id}[/dim]")
                continue
            if user_input.lower() == "/state":
                await _print_state(mind)
                continue

            burst = await mind.chat(conv_id, user_input)
            for msg in burst.messages:
                console.print(f"[bold blue]AI:[/bold blue] {msg}")
                await asyncio.sleep(config.burst_delay_ms / 1000.0)
            console.print()

    except (EOFError, KeyboardInterrupt):
        console.print()
    finally:
        await mind.stop()


async def _print_state(mind: FreudianMind):
    impressions = await mind.state.get_active_impressions()
    promotions = await mind.state.get_active_promotions()
    health = mind.defense_profile.get_health_report()

    console.print(Panel("[bold]Mind State[/bold]", border_style="yellow"))

    console.print(f"[yellow]Impressions ({len(impressions)}):[/yellow]")
    for imp in impressions[:8]:
        bar = "#" * int(imp["pressure"] * 10)
        marker = " <-- THRESHOLD" if imp["pressure"] >= mind.config.pressure_threshold else ""
        console.print(f"  [{imp['type'][:4]}] {bar} {imp['pressure']:.2f}{marker} | {imp['content'][:50]}")

    console.print(f"\n[green]Promotions ({len(promotions)}):[/green]")
    for p in promotions:
        console.print(f"  [{p['type']}] {p['key']}")

    console.print(f"\n[blue]Health:[/blue] maturity={health['maturity_score']} "
                  f"flexibility={health['flexibility_score']} "
                  f"growth={health['growth_velocity']}")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
