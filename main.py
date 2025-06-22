#!/usr/bin/env python3

import click
from memory_system import MemorySystem

@click.group()
def cli():
    """LLM Notepad - Your AI-powered second brain."""
    pass

@cli.command()
@click.argument('text')
@click.option('--tag', multiple=True, help='Add tags to the memory')
def store(text, tag):
    """Store a new memory."""
    memory_system = MemorySystem()
    
    metadata = {}
    if tag:
        metadata['tags'] = list(tag)
    
    memory_ids = memory_system.store_memory(text, metadata)
    click.echo(f"Stored memory in {len(memory_ids)} chunks with IDs: {', '.join(memory_ids[:3])}{'...' if len(memory_ids) > 3 else ''}")

@cli.command()
@click.argument('query')
@click.option('--raw', is_flag=True, help='Show raw memory search results instead of LLM response')
def query(query, raw):
    """Query your memories."""
    memory_system = MemorySystem()
    
    if raw:
        memories = memory_system.search_memories(query)
        click.echo("Relevant memories:")
        for i, memory in enumerate(memories, 1):
            click.echo(f"\n{i}. {memory['content']}")
            if memory.get('metadata', {}).get('tags'):
                click.echo(f"   Tags: {', '.join(memory['metadata']['tags'])}")
    else:
        response = memory_system.query_with_context(query)
        click.echo(response)

@cli.command()
def interactive():
    """Start interactive mode."""
    memory_system = MemorySystem()
    click.echo("LLM Notepad Interactive Mode")
    click.echo("Commands: /store <text>, /quit")
    click.echo("-" * 40)
    
    while True:
        user_input = click.prompt("", prompt_suffix="> ", default="", show_default=False)
        
        if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
            break
        elif user_input.startswith('/store '):
            text = user_input[7:]  # Remove '/store '
            memory_ids = memory_system.store_memory(text)
            click.echo(f"✓ Stored memory ({len(memory_ids)} chunks)")
        elif user_input.strip():
            response = memory_system.query_with_context(user_input)
            click.echo(f"\n{response}\n")

if __name__ == '__main__':
    cli()