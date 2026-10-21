import ast
import json
import math
import operator
import os
import sys
from datetime import date

import anthropic

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
MAX_TURNS = 10

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "abs": abs, "round": round, "min": min, "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*(_eval_node(a) for a in node.args))
    raise ValueError(f"Nepodporovaný výraz: {ast.dump(node)}")


def calculator(expression):
    result = _eval_node(ast.parse(expression, mode="eval"))
    return {"expression": expression, "result": result}


def days_between(start, end):
    d = (date.fromisoformat(end) - date.fromisoformat(start)).days
    return {"start": start, "end": end, "days": d}


TOOL_FUNCTIONS = {
    "calculator": calculator,
    "days_between": days_between,
}

TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Přesně vyhodnotí matematický výraz. Podporuje + - * / // % **, "
            "závorky, funkce sqrt, sin, cos, tan, log, log10, exp, abs, round, "
            "min, max a konstanty pi, e. Použij vždy, když je potřeba počítat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Výraz v syntaxi Pythonu, např. '(3 + 4) * sqrt(16)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "days_between",
        "description": "Vrátí počet dní mezi dvěma daty (end - start).",
        "input_schema": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "Datum YYYY-MM-DD"},
                "end": {"type": "string", "description": "Datum YYYY-MM-DD"},
            },
            "required": ["start", "end"],
        },
    },
]


def run_tool(name, tool_input):
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return json.dumps({"error": f"Neznámý nástroj: {name}"}), True
    try:
        return json.dumps(func(**tool_input), ensure_ascii=False), False
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False), True


def ask(question, client=None, verbose=True):
    client = client or anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system="Jsi užitečný asistent. Na výpočty vždy používej nástroje, nepočítej z hlavy.",
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output, is_error = run_tool(block.name, block.input)
            if verbose:
                print(f"[nástroj] {block.name}({block.input}) -> {output}", file=sys.stderr)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError("Překročen maximální počet kol konverzace.")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Kolik je 17 % z 2450 a jaká je odmocnina z výsledku?"
    print(ask(q))
