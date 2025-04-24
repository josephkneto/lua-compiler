import sys
from abc import abstractmethod


class Token():
    def __init__(self, type, value):
        self.type = type
        self.value = value


class PrePro():
    def filter(self, source):
        formattedText = ""
        i = 0
        while i < len(source):
            if source[i] == '-':
                if i + 1 < len(source) and source[i + 1] == '-':
                    allLineCommented = False
                    if source[i - 1] == '\n':
                        allLineCommented = True
                    i += 1
                    while i < len(source) and source[i] != '\n':
                        i += 1
                    if not allLineCommented:
                        formattedText += "\n"
                else:
                    formattedText += source[i]
            elif source[i] == '\n':
                if i != 0:
                    formattedText += source[i]
                while i + 1 < len(source) and source[i + 1] == '\n':
                    i += 1
            else:
                formattedText += source[i]
            i += 1

        return formattedText


class CodeGen():
    def __init__(self):
        self.code = '''SYS_EXIT equ 1
SYS_READ equ 3
SYS_WRITE equ 4
STDIN equ 0
STDOUT equ 1
True equ 1
False equ 0

segment .data

formatin: db "%d", 0
formatout: db "%d", 10, 0 ; newline, nul terminator
scanint: times 4 db 0 ; 32-bits integer = 4 bytes

segment .bss  ; variaveis
res RESB 1

section .text
global main ; linux
;global _main ; windows
extern scanf ; linux
extern printf ; linux
;extern _scanf ; windows
;extern _printf; windows
extern fflush ; linux
;extern _fflush ; windows
extern stdout ; linux
;extern _stdout ; windows

; subrotinas if/while
binop_je:
JE binop_true
JMP binop_false

binop_jg:
JG binop_true
JMP binop_false

binop_jl:
JL binop_true
JMP binop_false

binop_false:
MOV EAX, False  
JMP binop_exit
binop_true:
MOV EAX, True
binop_exit:
RET

main:

PUSH EBP ; guarda o base pointer
MOV EBP, ESP ; estabelece um novo base pointer
'''

    def writeCode(self, line):
        if not isinstance(line, str):
            line = str(line)
        self.code += line + "\n"

    def writeToFile(self, fileName):
        self.writeCode('''PUSH DWORD [stdout]
CALL fflush
ADD ESP, 4

MOV ESP, EBP
POP EBP

MOV EAX, 1
XOR EBX, EBX
INT 0x80''')
        with open(fileName, 'w') as f:
            f.write(self.code)


codeGen = CodeGen()


class Node():
    def __init__(self, value, children):
        self.value = value
        self.children = children

    @abstractmethod
    def Evaluate(self):
        pass


class BinOp(Node):
    def __init__(self, value, children):
        super().__init__(value, children)

    def Evaluate(self, st):
        rightValue = self.children[1].Evaluate(st)[0]
        codeGen.writeCode('''PUSH EAX''')
        leftValue = self.children[0].Evaluate(st)[0]
        codeGen.writeCode('''POP EBX''')
        if self.value == '+':
            codeGen.writeCode('''ADD EAX, EBX''')
            return [leftValue + rightValue, 'int']
        elif self.value == '-':
            codeGen.writeCode('''SUB EAX, EBX''')
            return [leftValue - rightValue, 'int']
        elif self.value == '*':
            codeGen.writeCode('''IMUL EAX, EBX''')
            return [leftValue * rightValue, 'int']
        elif self.value == '/':
            codeGen.writeCode('''IDIV EBX''')
            return [leftValue / rightValue, 'int']


class UnOp(Node):
    def __init__(self, value, children):
        super().__init__(value, children)

    def Evaluate(self, st):
        val, typ = self.children[0].Evaluate(st)
        if self.value == '-':
            codeGen.writeCode('''NEG EAX''')
            return [-val, typ]
        elif self.value == '+':
            return [val, typ]
        elif self.value == 'NOT':
            codeGen.writeCode('''NOT EAX''')
            return [not val, typ]


class IntVal(Node):
    def __init__(self, value):
        super().__init__(value, [])

    def Evaluate(self, st):
        codeGen.writeCode(f"MOV EAX, {int(self.value)}")
        return [self.value, 'int']


class NoOp(Node):
    def __init__(self):
        super().__init__(None, [])

    def Evaluate(self, st):
        pass


class Block(Node):
    def __init__(self, children):
        super().__init__(None, children)

    def Evaluate(self, st):
        for child in self.children:
            child.Evaluate(st)


class Identifier(Node):
    def __init__(self, value):
        super().__init__(value, [])

    def Evaluate(self, st):
        codeGen.writeCode(f"MOV EAX, [EBP - {st.getter(self.value)[2]}]")
        return st.getter(self.value)


class Assigment(Node):
    def __init__(self, children):
        super().__init__(None, children)

    def Evaluate(self, st):
        identifier = self.children[0].value
        val, typ, *x = self.children[1].Evaluate(st)
        st.setter(identifier, val, typ)
        codeGen.writeCode(f"MOV [EBP - {st.getter(identifier)[2]}], EAX")


class DeclareVar(Node):
    def __init__(self, children):
        super().__init__(None, children)

    def Evaluate(self, st):
        st.create(self.children[0].value)
        if self.children[1] is not None:
            val, typ = self.children[1].Evaluate(st)
            st.setter(self.children[0].value, val, typ)
        else:
            st.setter(self.children[0].value, None, None)
        codeGen.writeCode("PUSH DWORD 0")


class Print(Node):
    def __init__(self, children):
        super().__init__(None, children)

    def Evaluate(self, st):
        val, *x = self.children[0].Evaluate(st)
        codeGen.writeCode('''PUSH EAX
PUSH formatout
CALL printf
ADD ESP, 8''')
        print(val)


class BooleanOp(Node):
    def __init__(self, value, children):
        super().__init__(value, children)

    def Evaluate(self, st):
        rightValue, typ = self.children[1].Evaluate(st)
        codeGen.writeCode('PUSH EAX')
        leftValue, typ = self.children[0].Evaluate(st)
        codeGen.writeCode('POP EBX')
        if self.value == 'AND':
            codeGen.writeCode('AND EAX, EBX')
            return [leftValue and rightValue, typ]
        elif self.value == 'OR':
            codeGen.writeCode('OR EAX, EBX')
            return [leftValue or rightValue, typ]


class RelationalOp(Node):
    def __init__(self, value, children):
        super().__init__(value, children)

    def Evaluate(self, st):
        rightValue = self.children[1].Evaluate(st)[0]
        codeGen.writeCode('PUSH EAX')
        leftValue = self.children[0].Evaluate(st)[0]
        codeGen.writeCode('POP EBX')
        codeGen.writeCode('CMP EAX, EBX')
        if self.value == 'LT':
            codeGen.writeCode('CALL binop_jl')
            return [int(leftValue) < int(rightValue), 'int']
        elif self.value == 'GT':
            codeGen.writeCode('CALL binop_jg')
            return [int(leftValue) > int(rightValue), 'int']
        elif self.value == 'EQ':
            codeGen.writeCode('CALL binop_je')
            return [int(leftValue) == int(rightValue), 'int']
        codeGen.writeCode('MOVZX EAX, AL')


class IfStatement(Node):
    def __init__(self, children):
        super().__init__(None, children)

    def Evaluate(self, st):
        cond, x = self.children[0].Evaluate(st)
        if cond:
            for c in self.children[1]:
                c.Evaluate(st)
        elif len(self.children) > 2:
            for c in self.children[2]:
                c.Evaluate(st)

    def toAssembly(self):
        labelElse = f"ELSE_{Node.newId()}"
        labelEndIf = f"END_IF_{Node.newId()}"

        cond = self.children[0].toAssembly()
        codeGen.writeCode(f'''{cond}
CMP EAX, 0
JE {labelElse}''')

        ifPart = '\n'.join([c.generate_assembly() for c in self.children[1]])
        codeGen.writeCode(f'''{ifPart}
JMP {labelEndIf}''')

        if len(self.children) > 2:
            codeGen.writeCode(f"{labelElse}:")
            elsePart = '\n'.join([c.generate_assembly() for c in self.children[2]])
            codeGen.writeCode(f'''{elsePart}
{labelEndIf}:''')


class WhileLoop(Node):
    whileIdCounter = 0

    def __init__(self, children):
        super().__init__(None, children)
        self.whileId = WhileLoop.genId()

    def genId():
        WhileLoop.whileIdCounter += 1
        return WhileLoop.whileIdCounter

    def Evaluate(self, st):
        labelLoop = f"LOOP_{self.whileId}"
        labelExit = f"EXIT_{self.whileId}"
        codeGen.writeCode(f"{labelLoop}:")
        self.children[0].Evaluate(st)
        codeGen.writeCode(f'''CMP EAX, False
JE {labelExit}''')
        while self.children[0].Evaluate(st)[0]:
            for child in self.children[1]:
                result = child.Evaluate(st)
        codeGen.writeCode(f'''JMP {labelLoop}
{labelExit}:''')


class TerminalRead(Node):
    def __init__(self, children):
        super().__init__(None, children)

    def Evaluate(self, st):
        codeGen.writeCode('''PUSH scanint
PUSH formatin
call scanf
ADD ESP, 8
MOV EAX, DWORD [scanint]''')
        return [int(self.children[0]), 'int']


class SimbleTable():
    def __init__(self):
        self.st = {}
        self.shift = 0

    def getter(self, key):
        if key in self.st.keys():
            return self.st[key]
        else:
            raise Exception('Variable does not exist')

    def setter(self, key, value, type):
        if key in self.st.keys():
            self.st[key][0] = value
            self.st[key][1] = type
        else:
            raise Exception('Variable does not exist')

    def create(self, key):
        self.shift += 4
        if key in self.st:
            raise Exception('Variable already exists')
        self.st[key] = [None, None, self.shift]


class Tokenizer():
    def __init__(self, source, position=0, next=None):
        no_comments_source = PrePro().filter(source)
        self.source = no_comments_source
        self.position = position
        self.next = next
        self.reserved = ['print', 'if', 'then', 'else', 'end', 'do', 'while', 'read', 'and', 'or', 'not', 'local']

    def selectNext(self):
        while self.position < len(self.source) and (self.source[self.position] == ' ' or self.source[self.position] == '\t'):
            self.position += 1

        if self.position < len(self.source):
            if self.source[self.position] == '+':
                self.next = Token('PLUS', '+')
                self.position += 1
            elif self.source[self.position] == '-':
                self.next = Token('MINUS', '-')
                self.position += 1
            elif self.source[self.position] == '*':
                self.next = Token('MULT', '*')
                self.position += 1
            elif self.source[self.position] == '/':
                self.next = Token('DIV', '/')
                self.position += 1
            elif self.source[self.position] == '(':
                self.next = Token('PAR_OPEN', '(')
                self.position += 1
            elif self.source[self.position] == ')':
                self.next = Token('PAR_CLOSE', ')')
                self.position += 1
            elif self.source[self.position] == '\n':
                self.next = Token('NEW_LINE', '\n')
                self.position += 1
            elif self.source[self.position].isalpha():
                identifier = ''
                while self.position < len(self.source) and (self.source[self.position].isalpha() or self.source[self.position].isdigit() or self.source[self.position] == '_'):
                    identifier += self.source[self.position]
                    self.position += 1
                if identifier in self.reserved:
                    self.next = Token(identifier.upper(), identifier)
                else:
                    self.next = Token('IDENTIFIER', identifier)
            elif self.source[self.position] == "=":
                self.position += 1
                if self.source[self.position] == "=":
                    self.next = Token('EQ', '==')
                    self.position += 1
                elif self.source[self.position] == '>':
                    self.next = Token('ASSIGN', '=>')
                    self.position += 1
                elif self.source[self.position] == '<':
                    self.next = Token('ASSIGN', '=<')
                    self.position += 1
                else:
                    self.next = Token('ASSIGN', '=')
            elif self.source[self.position] == '!':
                self.position += 1
                if self.source[self.position] == '=':
                    self.next = Token('NEQ', '!=')
                    self.position += 1
                else:
                    raise Exception('Unexpected token')
            elif self.source[self.position].isdigit():
                number = ''
                while self.position < len(self.source) and self.source[self.position].isdigit():
                    number += self.source[self.position]
                    self.position += 1
                self.next = Token('NUMBER', number)
            elif self.source[self.position] in ['<', '>']:
                token_type = {'<': 'LT', '>': 'GT'}[self.source[self.position]]
                self.next = Token(token_type, self.source[self.position])
                self.position += 1
            elif self.source[self.position] == ".":
                if self.position + 1 < len(self.source) and self.source[self.position + 1] == ".":
                    self.next = Token("CONCAT", '..')
                    self.position += 2
            elif self.source[self.position] == '"':
                string = ""
                self.position += 1
                while self.source[self.position] != '"':
                    string += self.source[self.position]
                    self.position += 1
                self.next = Token('STRING', string)
                self.position += 1
            else:
                raise Exception('Unexpected token')
        else:
            self.next = None


class Parser():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def parseBlock(self):
        children = []
        while self.tokenizer.next is not None:
            children.append(self.parseStatement())
        return Block(children)

    def parseStatement(self):
        if self.tokenizer.next.type == 'IDENTIFIER':
            identifier = self.tokenizer.next.value
            self.tokenizer.selectNext()
            if self.tokenizer.next.type == 'ASSIGN':
                self.tokenizer.selectNext()
                toAssign = self.parseBoolExpression()
                assigment = Assigment([Identifier(identifier), toAssign])
                if self.tokenizer.next.type != 'NEW_LINE':
                    raise Exception('Expected \\n')
                self.tokenizer.selectNext()
                return assigment
            else:
                raise Exception('Unexpected token')
        elif self.tokenizer.next.type == 'LOCAL':
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'IDENTIFIER':
                raise Exception('Unexpected token')
            identifier = self.tokenizer.next.value
            self.tokenizer.selectNext()
            if self.tokenizer.next.type == 'ASSIGN':
                self.tokenizer.selectNext()
                result = self.parseBoolExpression()
                self.tokenizer.selectNext()
                return DeclareVar([Identifier(identifier), result])
            elif self.tokenizer.next.type == 'NEW_LINE':
                self.tokenizer.selectNext()
                return DeclareVar([Identifier(identifier), None])
        elif self.tokenizer.next.type == 'PRINT':
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'PAR_OPEN':
                raise Exception('Unexpected token')
            self.tokenizer.selectNext()
            expression = self.parseBoolExpression()
            if self.tokenizer.next.type != 'PAR_CLOSE':
                raise Exception('Unexpected token')
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'NEW_LINE':
                raise Exception('Expected \\n')
            self.tokenizer.selectNext()
            return Print([expression])
        elif self.tokenizer.next.type == 'IF':
            self.tokenizer.selectNext()
            condition = self.parseBoolExpression()
            if self.tokenizer.next.type != 'THEN':
                raise Exception('Unexpected token')
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'NEW_LINE':
                raise Exception('Expected \\n')
            self.tokenizer.selectNext()
            if_block = []
            while self.tokenizer.next.type != 'ELSE' and self.tokenizer.next.type != 'END':
                statement = self.parseStatement()
                if_block.append(statement)
            if self.tokenizer.next.type != 'ELSE' and self.tokenizer.next.type != 'END':
                raise Exception('Expected else or end\n')
            if self.tokenizer.next.type == 'ELSE':
                self.tokenizer.selectNext()
                if self.tokenizer.next.type != 'NEW_LINE':
                    raise Exception('Expected \\n')
                self.tokenizer.selectNext()
                else_block = []
                while self.tokenizer.next.type != 'END':
                    statement = self.parseStatement()
                    else_block.append(statement)
                if self.tokenizer.next.type != 'END':
                    raise Exception('Expected end\n')
                self.tokenizer.selectNext()
                if self.tokenizer.next.type != 'NEW_LINE' and self.tokenizer.next != None:
                    raise Exception("Expected \\n\n")
                return IfStatement([condition, if_block, else_block])
            if self.tokenizer.next.type == 'END':
                self.tokenizer.selectNext()
                if self.tokenizer.next.type != 'NEW_LINE' and self.tokenizer.next != None:
                    raise Exception(f"Expected \\n\n")
                return IfStatement([condition, if_block, []])
        elif self.tokenizer.next.type == 'WHILE':
            self.tokenizer.selectNext()
            condition = self.parseBoolExpression()
            if self.tokenizer.next.type != 'DO':
                raise Exception('Unexpected token')
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'NEW_LINE':
                raise Exception('Expected \\n')
            self.tokenizer.selectNext()
            statements = []
            while self.tokenizer.next.type != 'END':
                statements.append(self.parseStatement())
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'NEW_LINE':
                raise Exception('Expected \\n')
            self.tokenizer.selectNext()
            return WhileLoop([condition, statements])
        elif self.tokenizer.next.type == 'NEW_LINE':
            self.tokenizer.selectNext()
            return NoOp()
        else:
            raise Exception(f'Unexpected token {self.tokenizer.next.value}')

    def parseTerm(self):
        result = self.parseFactor()
        while self.tokenizer.next is not None and (self.tokenizer.next.type == 'MULT' or self.tokenizer.next.type == 'DIV'):
            if self.tokenizer.next.type == 'MULT':
                self.tokenizer.selectNext()
                result = BinOp('*', [result, self.parseFactor()])
            elif self.tokenizer.next.type == 'DIV':
                self.tokenizer.selectNext()
                result = BinOp('/', [result, self.parseFactor()])
        return result

    def parseExpression(self):
        result = self.parseTerm()
        while self.tokenizer.next is not None and (self.tokenizer.next.type == 'PLUS' or self.tokenizer.next.type == 'MINUS' or self.tokenizer.next.type == 'CONCAT'):
            if self.tokenizer.next.type == 'PLUS':
                self.tokenizer.selectNext()
                result = BinOp('+', [result, self.parseTerm()])
            elif self.tokenizer.next.type == 'MINUS':
                self.tokenizer.selectNext()
                result = BinOp('-', [result, self.parseTerm()])
            elif self.tokenizer.next.type == 'CONCAT':
                self.tokenizer.selectNext()
                result = BinOp('..', [result, self.parseTerm()])
        return result

    def parseFactor(self):
        if self.tokenizer.next.type == 'NUMBER':
            number = int(self.tokenizer.next.value)
            self.tokenizer.selectNext()
            return IntVal(number)
        elif self.tokenizer.next is not None and self.tokenizer.next.type == 'PLUS':
            self.tokenizer.selectNext()
            return UnOp('+', [self.parseFactor()])
        elif self.tokenizer.next is not None and self.tokenizer.next.type == 'MINUS':
            self.tokenizer.selectNext()
            return UnOp('-', [self.parseFactor()])
        elif self.tokenizer.next is not None and self.tokenizer.next.type == 'NOT':
            self.tokenizer.selectNext()
            return UnOp('NOT', [self.parseFactor()])
        elif self.tokenizer.next is not None and self.tokenizer.next.type == 'PAR_OPEN':
            self.tokenizer.selectNext()
            number = self.parseBoolExpression()
            if self.tokenizer.next.type != 'PAR_CLOSE':
                raise Exception('Unexpected token')
            self.tokenizer.selectNext()
            return number
        elif self.tokenizer.next.type == 'IDENTIFIER':
            identifier = self.tokenizer.next.value
            self.tokenizer.selectNext()
            return Identifier(identifier)
        elif self.tokenizer.next.type == 'READ':
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'PAR_OPEN':
                raise Exception('Unexpected token')
            self.tokenizer.selectNext()
            if self.tokenizer.next.type != 'PAR_CLOSE':
                raise Exception('Unexpected token')
            self.tokenizer.selectNext()
            return TerminalRead([int(input())])
        else:
            raise Exception('Unexpected token')

    def parseBoolExpression(self):
        result = self.parseBoolTerm()
        while self.tokenizer.next is not None and self.tokenizer.next.type == 'OR':
            self.tokenizer.selectNext()
            result = BooleanOp('OR', [result, self.parseBoolTerm()])
        return result

    def parseBoolTerm(self):
        result = self.paseRelExpression()
        while self.tokenizer.next is not None and self.tokenizer.next.type == 'AND':
            self.tokenizer.selectNext()
            result = BooleanOp('AND', [result, self.paseRelExpression()])
        return result

    def paseRelExpression(self):
        result = self.parseExpression()
        if self.tokenizer.next.type in ['LT', 'GT', 'EQ', 'NEQ', 'LTE', 'GTE']:
            token_type = self.tokenizer.next.type
            self.tokenizer.selectNext()
            return RelationalOp(token_type, [result, self.parseExpression()])
        else:
            return result

    def run(self, code, st):
        self.tokenizer = Tokenizer(code)
        self.tokenizer.selectNext()
        result = self.parseBlock()
        result = result.Evaluate(st)
        if self.tokenizer.next is not None:
            raise Exception('Unexpected token')
        return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py 'expression'")
        sys.exit(1)

    file = sys.argv[1]

    if not file.endswith(".lua"):
        print("Invalid file extension")
        sys.exit(1)

    with open(file, 'r') as f:
        expression = f.read()

    tokenizer = Tokenizer(expression)
    parser = Parser(tokenizer)
    parser.run(expression, SimbleTable())

    codeGen.writeToFile(file.replace(".lua", ".asm"))
