
# 🧠 Lua-Compiler

**Lua-Compiler** is a simple compiler project built during university coursework. It implements a custom compiler that parses a restricted subset of the Lua programming language and generates equivalent **x86 assembly code**. This educational project showcases core concepts of **lexical analysis**, **parsing**, **semantic evaluation**, and **code generation**.

---

## 🛠️ Features

- 🔤 **Tokenizer (Lexer)**: Identifies reserved keywords, operators, identifiers, literals, and more.
- 🌳 **Parser**: Builds an abstract syntax tree (AST) from Lua-like syntax.
- 🧠 **Semantic Analysis & Evaluation**: Supports integer operations, boolean expressions, control flow (`if`, `while`), variable declarations, assignments, and printing.
- 🧾 **Code Generation**: Outputs NASM-compatible x86 assembly code, complete with data sections and function calls like `printf` and `scanf`.
- 🎯 **Test File**: Includes a test Lua file (`teste.lua`) to demonstrate supported syntax and capabilities.

---

## 🚀 Getting Started

### ✅ Prerequisites
- Python 3.x
- NASM (Netwide Assembler)
- GCC (or another linker to assemble and run generated `.asm` code)

### ▶️ How to Run

1. **Compile Lua-like code to Assembly**
   ```bash
   python main.py teste.lua
   ```

2. **Assemble the generated `.asm` file**
   ```bash
   nasm -f elf32 teste.asm -o teste.o
   gcc -m32 -o teste teste.o
   ./teste
   ```

---

## 📂 File Structure

- `main.py`: Core compiler implementation, including tokenizer, parser, AST classes, and code generator.
- `teste.lua`: Sample Lua-like script used for testing the compiler.
- `teste.asm`: Output assembly file generated from `teste.lua`.

---

## 📚 Example Syntax

```lua
local x
local y
local z = "x: "
x = 1
y = x or (1 == 1)
print(x + y)
print(z .. x)
```

---

## 📎 Notes

- This compiler handles a simplified Lua syntax, including logical and arithmetic expressions, but does not support functions or full type checking.
- The assembly is tailored for Linux systems using 32-bit architecture (NASM syntax).
- Designed for educational purposes only.

---
