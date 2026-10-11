import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QPlainTextEdit, QLabel, 
                               QComboBox, QSlider, QGroupBox, QGridLayout)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QFont

class CACpu:
    """
    16/32/64-bit CPU evaluated as 1D Cellular Automata.
    Expanded ISA: 8 Registers, Signed Control Flow, Block Memory, Jump Tables.
    """
    def __init__(self, bit_width=32):
        self.set_bit_width(bit_width)
        self.reset()

    def set_bit_width(self, bit_width):
        self.bit_width = bit_width
        self.mask = (1 << bit_width) - 1
        self.sign_bit = 1 << (bit_width - 1)

    def reset(self):
        self.pc = 0
        self.sp = 32       # Stack starts higher to accommodate 8 regs & data
        self.regs = [0] * 8 # R0 to R7
        self.memory = [0] * 256
        self.program = []
        self.labels = {}
        self.halted = False
        self.last_instr = "NONE"
        self.flags = {'Z': 0, 'C': 0, 'N': 0, 'V': 0}
        self.active_mem = -1
        self.active_reg = -1

    def load_program(self, code):
        self.reset()
        lines = code.strip().split('\n')
        clean_lines = []
        temp_pc = 0
        
        for line in lines:
            if ';' in line: line = line.split(';')[0]
            line = line.strip()
            if not line: continue
            
            if line.endswith(':'):
                self.labels[line[:-1].strip()] = temp_pc
            else:
                if ':' in line:
                    parts = line.split(':')
                    self.labels[parts[0].strip()] = temp_pc
                    line = parts[1].strip()
                clean_lines.append(line)
                temp_pc += 1
                
        for line in clean_lines:
            parts = line.replace(',', ' ').split()
            for i in range(1, len(parts)):
                if parts[i] in self.labels:
                    parts[i] = str(self.labels[parts[i]])
            self.program.append(parts)

    def to_binary_list(self, val):
        return [(val >> i) & 1 for i in range(self.bit_width)]

    def from_binary_list(self, bits):
        val = 0
        for i in range(len(bits)):
            if bits[i]: val |= (1 << i)
        return val

    def ca_add(self, a, b):
        bits_a = self.to_binary_list(a)
        bits_b = self.to_binary_list(b)
        result_bits = []
        carry = 0
        for i in range(self.bit_width):
            a_bit, b_bit = bits_a[i], bits_b[i]
            sum_bit = a_bit ^ b_bit ^ carry
            carry = (a_bit & b_bit) | (carry & (a_bit ^ b_bit))
            result_bits.append(sum_bit)
        res = self.from_binary_list(result_bits)
        
        self.flags['C'] = carry
        self.flags['Z'] = 1 if res == 0 else 0
        self.flags['N'] = 1 if (res & self.sign_bit) else 0
        # Signed Overflow: if sign of a and b are same, but sign of res is different
        sa = (a & self.sign_bit) != 0
        sb = (b & self.sign_bit) != 0
        sr = (res & self.sign_bit) != 0
        self.flags['V'] = 1 if (sa == sb and sr != sa) else 0
        return res

    def ca_mul(self, a, b):
        res = 0
        while b > 0:
            if b & 1: res = self.ca_add(res, a)
            a = (a << 1) & self.mask
            b = b >> 1
        self.flags['Z'] = 1 if res == 0 else 0
        return res

    def ca_div_mod(self, a, b):
        if b == 0:
            self.flags['Z'] = 1
            return 0, 0
        quotient = 0
        remainder = 0
        for i in range(self.bit_width - 1, -1, -1):
            remainder = (remainder << 1) | ((a >> i) & 1)
            remainder &= self.mask
            if remainder >= b:
                remainder = self.ca_add(remainder, self.ca_add(self.ca_logic(b, 0, 'NOT'), 1))
                quotient |= (1 << i)
        self.flags['Z'] = 1 if quotient == 0 else 0
        return quotient, remainder

    def ca_logic(self, a, b, op):
        bits_a = self.to_binary_list(a)
        bits_b = self.to_binary_list(b)
        result_bits = []
        for i in range(self.bit_width):
            if op == 'AND': result_bits.append(bits_a[i] & bits_b[i])
            elif op == 'OR': result_bits.append(bits_a[i] | bits_b[i])
            elif op == 'XOR': result_bits.append(bits_a[i] ^ bits_b[i])
            elif op == 'NAND': result_bits.append(1 - (bits_a[i] & bits_b[i]))
            elif op == 'NOR': result_bits.append(1 - (bits_a[i] | bits_b[i]))
            elif op == 'XNOR': result_bits.append(1 - (bits_a[i] ^ bits_b[i]))
            elif op == 'NOT': result_bits.append(1 - bits_a[i])
        res = self.from_binary_list(result_bits)
        self.flags['Z'] = 1 if res == 0 else 0
        self.flags['N'] = 1 if (res & self.sign_bit) else 0
        return res

    def ca_rotate(self, val, direction):
        if direction == 'LEFT':
            msb = (val >> (self.bit_width - 1)) & 1
            val = ((val << 1) | msb) & self.mask
        elif direction == 'RIGHT':
            lsb = val & 1
            val = ((val >> 1) | (lsb << (self.bit_width - 1))) & self.mask
        self.flags['C'] = 1 if (direction == 'LEFT' and msb) or (direction == 'RIGHT' and lsb) else 0
        return val

    def get_reg_val(self, token):
        if token.startswith('R') and len(token) == 2:
            idx = int(token[1])
            self.active_reg = idx
            return self.regs[idx]
        return int(token) & self.mask

    def set_reg_val(self, token, val):
        if token.startswith('R') and len(token) == 2:
            idx = int(token[1])
            self.regs[idx] = val & self.mask
            self.active_reg = idx

    def step(self):
        self.active_mem = -1
        self.active_reg = -1
        if self.halted or self.pc >= len(self.program):
            self.halted = True
            self.last_instr = "HALTED"
            return

        instr = self.program[self.pc]
        opcode = instr[0].upper()
        self.last_instr = f"PC:{self.pc: <3} | {' '.join(instr): <24}"
        self.pc += 1

        try:
            if opcode == 'NOP': pass
            elif opcode == 'SET': self.set_reg_val(instr[1], int(instr[2]))
            elif opcode == 'SETM': 
                self.memory[int(instr[1])] = int(instr[2]) & self.mask
                self.active_mem = int(instr[1])
            elif opcode == 'MOV': self.set_reg_val(instr[1], self.get_reg_val(instr[2]))
            elif opcode == 'SWAP':
                v1 = self.get_reg_val(instr[1]); v2 = self.get_reg_val(instr[2])
                self.set_reg_val(instr[1], v2); self.set_reg_val(instr[2], v1)
            elif opcode == 'CLR': self.set_reg_val(instr[1], 0)
            
            # Math
            elif opcode == 'ADD': self.set_reg_val(instr[1], self.ca_add(self.get_reg_val(instr[1]), self.get_reg_val(instr[2])))
            elif opcode == 'ADDI': self.set_reg_val(instr[1], self.ca_add(self.get_reg_val(instr[1]), int(instr[2]) & self.mask))
            elif opcode == 'SUB':
                b = self.get_reg_val(instr[2])
                self.set_reg_val(instr[1], self.ca_add(self.get_reg_val(instr[1]), self.ca_add(self.ca_logic(b, 0, 'NOT'), 1)))
            elif opcode == 'SUBI':
                b = int(instr[2]) & self.mask
                self.set_reg_val(instr[1], self.ca_add(self.get_reg_val(instr[1]), self.ca_add(self.ca_logic(b, 0, 'NOT'), 1)))
            elif opcode == 'MUL': self.set_reg_val(instr[1], self.ca_mul(self.get_reg_val(instr[1]), self.get_reg_val(instr[2])))
            elif opcode == 'DIV':
                q, _ = self.ca_div_mod(self.get_reg_val(instr[1]), self.get_reg_val(instr[2]))
                self.set_reg_val(instr[1], q)
            elif opcode == 'MOD':
                _, r = self.ca_div_mod(self.get_reg_val(instr[1]), self.get_reg_val(instr[2]))
                self.set_reg_val(instr[1], r)
            elif opcode == 'INC': self.set_reg_val(instr[1], self.ca_add(self.get_reg_val(instr[1]), 1))
            elif opcode == 'DEC': self.set_reg_val(instr[1], self.ca_add(self.get_reg_val(instr[1]), self.ca_add(self.mask, 1)))
            elif opcode == 'NEG': 
                a = self.get_reg_val(instr[1])
                self.set_reg_val(instr[1], self.ca_add(self.ca_logic(a, 0, 'NOT'), 1))
            elif opcode == 'ABS': 
                a = self.get_reg_val(instr[1])
                if a & self.sign_bit: 
                    self.set_reg_val(instr[1], self.ca_add(self.ca_logic(a, 0, 'NOT'), 1))
            
            # Logic
            elif opcode in ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR']:
                self.set_reg_val(instr[1], self.ca_logic(self.get_reg_val(instr[1]), self.get_reg_val(instr[2]), opcode))
            elif opcode in ['ANDI', 'ORI', 'XORI']:
                op_map = {'ANDI': 'AND', 'ORI': 'OR', 'XORI': 'XOR'}
                self.set_reg_val(instr[1], self.ca_logic(self.get_reg_val(instr[1]), int(instr[2]) & self.mask, op_map[opcode]))
            elif opcode == 'NOT':
                self.set_reg_val(instr[1], self.ca_logic(self.get_reg_val(instr[1]), 0, 'NOT'))
            
            # Bit Manipulation
            elif opcode == 'BSET':
                val = self.get_reg_val(instr[1]); bit_idx = int(instr[2])
                self.set_reg_val(instr[1], val | (1 << bit_idx))
            elif opcode == 'BCLR':
                val = self.get_reg_val(instr[1]); bit_idx = int(instr[2])
                self.set_reg_val(instr[1], val & ~(1 << bit_idx))
            elif opcode == 'BTST':
                val = self.get_reg_val(instr[1]); bit_idx = int(instr[2])
                bit = (val >> bit_idx) & 1
                self.flags['Z'] = 0 if bit else 1
            elif opcode == 'CLZ': 
                a = self.get_reg_val(instr[1])
                count = 0
                for i in range(self.bit_width - 1, -1, -1):
                    if (a >> i) & 1: break
                    count += 1
                self.set_reg_val(instr[1], count)
            elif opcode == 'CTZ': 
                a = self.get_reg_val(instr[1])
                count = 0
                for i in range(0, self.bit_width):
                    if (a >> i) & 1: break
                    count += 1
                self.set_reg_val(instr[1], count)
            elif opcode == 'BITREV': 
                a = self.get_reg_val(instr[1])
                rev = 0
                for i in range(self.bit_width):
                    if (a >> i) & 1: rev |= (1 << (self.bit_width - 1 - i))
                self.set_reg_val(instr[1], rev)

            elif opcode == 'SHL': self.set_reg_val(instr[1], (self.get_reg_val(instr[1]) << 1) & self.mask)
            elif opcode == 'SHR': self.set_reg_val(instr[1], (self.get_reg_val(instr[1]) >> 1) & self.mask)
            elif opcode == 'ROL': self.set_reg_val(instr[1], self.ca_rotate(self.get_reg_val(instr[1]), 'LEFT'))
            elif opcode == 'ROR': self.set_reg_val(instr[1], self.ca_rotate(self.get_reg_val(instr[1]), 'RIGHT'))
            
            # Memory
            elif opcode == 'LOAD':
                self.set_reg_val(instr[1], self.memory[int(instr[2])])
                self.active_mem = int(instr[2])
            elif opcode == 'SAVE':
                self.memory[int(instr[1])] = self.get_reg_val(instr[2])
                self.active_mem = int(instr[1])
            elif opcode == 'LOADR':
                addr = self.get_reg_val(instr[2])
                self.set_reg_val(instr[1], self.memory[addr])
                self.active_mem = addr
            elif opcode == 'SAVER':
                addr = self.get_reg_val(instr[1])
                self.memory[addr] = self.get_reg_val(instr[2])
                self.active_mem = addr
            elif opcode == 'LOADO':
                addr = (self.get_reg_val(instr[2]) + int(instr[3])) & 0xFF
                self.set_reg_val(instr[1], self.memory[addr])
                self.active_mem = addr
            elif opcode == 'SAVEO':
                addr = (self.get_reg_val(instr[1]) + int(instr[2])) & 0xFF
                self.memory[addr] = self.get_reg_val(instr[3])
                self.active_mem = addr

            # Block Memory Ops
            elif opcode == 'MEMCPY': # MEMCPY Rdst, Rsrc, Rlen
                dst = self.get_reg_val(instr[1]); src = self.get_reg_val(instr[2]); length = self.get_reg_val(instr[3])
                for i in range(length):
                    self.memory[(dst + i) & 0xFF] = self.memory[(src + i) & 0xFF]
                    self.active_mem = (dst + i) & 0xFF
            elif opcode == 'MEMSET': # MEMSET Rdst, Rval, Rlen
                dst = self.get_reg_val(instr[1]); val = self.get_reg_val(instr[2]); length = self.get_reg_val(instr[3])
                for i in range(length):
                    self.memory[(dst + i) & 0xFF] = val
                    self.active_mem = (dst + i) & 0xFF
                
            elif opcode == 'PUSH':
                self.memory[self.sp] = self.get_reg_val(instr[1])
                self.active_mem = self.sp
                self.sp -= 1
            elif opcode == 'POP':
                self.sp += 1
                self.set_reg_val(instr[1], self.memory[self.sp])
                self.active_mem = self.sp
                
            elif opcode == 'CMOVZ': 
                if self.flags['Z']: self.set_reg_val(instr[1], self.get_reg_val(instr[2]))
            elif opcode == 'CMOVNZ': 
                if not self.flags['Z']: self.set_reg_val(instr[1], self.get_reg_val(instr[2]))
                
            elif opcode == 'CMP':
                b = self.get_reg_val(instr[2])
                self.ca_add(self.get_reg_val(instr[1]), self.ca_add(self.ca_logic(b, 0, 'NOT'), 1))
            elif opcode == 'CMPI':
                b = int(instr[2]) & self.mask
                self.ca_add(self.get_reg_val(instr[1]), self.ca_add(self.ca_logic(b, 0, 'NOT'), 1))
                
            # Flow Control
            elif opcode == 'JMP': self.pc = int(instr[1])
            elif opcode == 'JE': 
                if self.flags['Z']: self.pc = int(instr[1])
            elif opcode == 'JNE': 
                if not self.flags['Z']: self.pc = int(instr[1])
            elif opcode == 'JC': 
                if self.flags['C']: self.pc = int(instr[1])
            elif opcode == 'JNC': 
                if not self.flags['C']: self.pc = int(instr[1])
                
            # Signed Control Flow (Relies on N and V flags)
            elif opcode == 'JG':  # Signed >  (Z==0 and N==V)
                if not self.flags['Z'] and (self.flags['N'] == self.flags['V']): self.pc = int(instr[1])
            elif opcode == 'JGE': # Signed >= (N==V)
                if self.flags['N'] == self.flags['V']: self.pc = int(instr[1])
            elif opcode == 'JL':  # Signed <  (N!=V)
                if self.flags['N'] != self.flags['V']: self.pc = int(instr[1])
            elif opcode == 'JLE': # Signed <= (Z==1 or N!=V)
                if self.flags['Z'] or (self.flags['N'] != self.flags['V']): self.pc = int(instr[1])
                
            # Indirect Flow (Jump Tables)
            elif opcode == 'JMPR': self.pc = self.get_reg_val(instr[1])
            elif opcode == 'CALLR':
                self.memory[self.sp] = self.pc
                self.active_mem = self.sp
                self.sp -= 1
                self.pc = self.get_reg_val(instr[1])
            elif opcode == 'CALL':
                self.memory[self.sp] = self.pc
                self.active_mem = self.sp
                self.sp -= 1
                self.pc = int(instr[1])
            elif opcode == 'BL': 
                self.regs[7] = self.pc
                self.active_reg = 7
                self.pc = int(instr[1])
            elif opcode == 'RET': # RET n (optional stack cleanup)
                self.sp += 1
                self.pc = self.memory[self.sp]
                self.active_mem = self.sp
                if len(instr) > 1: # RET n
                    self.sp += int(instr[1])
            elif opcode == 'HALT':
                self.halted = True
                self.pc -= 1
        except Exception as e:
            self.last_instr += f" [ERROR: {e}]"
            self.halted = True


class CAHistoryWidget(QWidget):
    def __init__(self, bit_width, height=16, cell_size=8, highlight_active=False):
        super().__init__()
        self.bit_width = bit_width
        self.height = height
        self.cell_size = cell_size
        self.history = [[0] * bit_width for _ in range(height)]
        self.is_active = False
        self.highlight_active = highlight_active
        self.overlay_text = ""
        self.update_geometry()

    def reset_history(self, bit_width):
        self.bit_width = bit_width
        self.history = [[0] * bit_width for _ in range(self.height)]
        self.update_geometry()
        self.update()

    def update_geometry(self):
        self.setFixedSize(self.bit_width * self.cell_size + 2, self.height * self.cell_size + 2)

    def push_state(self, val, text=""):
        self.history.pop(0)
        self.history.append([(val >> i) & 1 for i in range(self.bit_width)])
        self.overlay_text = text
        self.update()

    def flash_active(self):
        self.is_active = True
        self.update()
        QTimer.singleShot(200, self.clear_flash)

    def clear_flash(self):
        self.is_active = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        
        if self.is_active and self.highlight_active:
            painter.setBrush(QColor(255, 150, 0)) 
            painter.drawRect(0, 0, self.width(), self.height())

        for y, row in enumerate(self.history):
            alpha = int(255 * ((y + 1) / self.height))
            for x, bit in enumerate(row):
                if bit == 1:
                    painter.setBrush(QColor(0, 255, 100, alpha))
                else:
                    painter.setBrush(QColor(25, 25, 25, alpha))
                painter.drawRect(x * self.cell_size + 1, y * self.cell_size + 1, self.cell_size, self.cell_size)
        
        if self.overlay_text:
            painter.setPen(QColor(255, 255, 255, 180))
            painter.setFont(QFont("Courier", 10, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, self.overlay_text)


class LEDWidget(QWidget):
    def __init__(self, color=QColor(0, 255, 0)):
        super().__init__()
        self.color = color
        self.is_on = False
        self.setFixedSize(20, 20)

    def set_on(self, state):
        self.is_on = state
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.is_on:
            painter.setBrush(self.color)
        else:
            painter.setBrush(QColor(40, 40, 40))
        painter.setPen(QColor(90, 90, 90))
        painter.drawEllipse(2, 2, 16, 16)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced 1D CA Computer (8 Regs, Signed Flow, Block Ops)")
        self.resize(1300, 900)
        self.cpu = CACpu(bit_width=32)
        self.base_format = 'hex'
        self.init_ui()
        self.load_example_program(0)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QVBoxLayout()
        
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Arch:"))
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["16-bit", "32-bit", "64-bit"])
        self.arch_combo.setCurrentIndex(1)
        self.arch_combo.currentIndexChanged.connect(self.change_architecture)
        ctrl_layout.addWidget(self.arch_combo)
        
        ctrl_layout.addWidget(QLabel("Example:"))
        self.example_combo = QComboBox()
        self.example_combo.addItems(["Countdown", "Fibonacci", "Bit Rotation", "Array Sum", "Jump Table", "Block Copy"])
        self.example_combo.currentIndexChanged.connect(self.load_example_program)
        ctrl_layout.addWidget(self.example_combo)
        left_panel.addLayout(ctrl_layout)

        self.code_edit = QPlainTextEdit()
        self.code_edit.setFont(QFont("Courier", 11))
        left_panel.addWidget(self.code_edit)

        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Assemble & Load")
        self.btn_load.clicked.connect(self.load_program)
        self.btn_step = QPushButton("Step")
        self.btn_step.clicked.connect(self.step_cpu)
        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self.run_cpu)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_cpu)
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_step)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_reset)
        left_panel.addLayout(btn_layout)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Clock Delay (ms):"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 1000)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.update_clock_speed)
        speed_layout.addWidget(self.speed_slider)
        left_panel.addLayout(speed_layout)

        log_group = QGroupBox("Execution Log")
        log_layout = QVBoxLayout()
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(QFont("Courier", 10))
        self.log_console.setMaximumHeight(120)
        log_layout.addWidget(self.log_console)
        log_group.setLayout(log_layout)
        left_panel.addWidget(log_group)

        isa_group = QGroupBox("ISA Reference")
        isa_layout = QVBoxLayout()
        isa_text = QPlainTextEdit()
        isa_text.setReadOnly(True)
        isa_text.setFont(QFont("Courier", 9))
        isa_text.setPlainText(
            "DATA:\n"
            "  SET Rd, val | SETM addr, val\n"
            "  MOV Rd, Rs  | SWAP Rd, Rs | CLR Rd\n"
            "  LOAD Rd, addr | SAVE addr, Rs\n"
            "  LOADR Rd, Rptr | SAVER Rptr, Rs\n"
            "  LOADO Rd, Rbase, off | SAVEO Rbase, off, Rs\n"
            "  PUSH Rs | POP Rd\n"
            "BLOCK MEMORY:\n"
            "  MEMCPY Rdst, Rsrc, Rlen\n"
            "  MEMSET Rdst, Rval, Rlen\n"
            "MATH & SIGNED:\n"
            "  ADD/SUB Rd, Rs | ADDI/SUBI Rd, val\n"
            "  MUL | DIV | MOD Rd, Rs\n"
            "  NEG Rd | ABS Rd\n"
            "  INC Rd | DEC Rd | NOP\n"
            "LOGIC & BITS:\n"
            "  AND|OR|XOR|NAND|NOR|XNOR Rd, Rs\n"
            "  ANDI|ORI|XORI Rd, val\n"
            "  NOT Rd | SHL Rd | SHR Rd\n"
            "  ROL Rd | ROR Rd | BITREV Rd\n"
            "  BSET/BCLR/BTST Rd, i\n"
            "  CLZ Rd | CTZ Rd\n"
            "FLOW & BRANCHLESS:\n"
            "  CMP Rd, Rs | CMPI Rd, val\n"
            "  CMOVZ/CMOVNZ Rd, Rs\n"
            "  JMP label | JE | JNE | JC | JNC\n"
            "  JG | JGE | JL | JLE (Signed)\n"
            "  JMPR Rd | CALL label | CALLR Rd\n"
            "  BL label (Link R7) | RET | RET n\n"
            "  HALT"
        )
        isa_layout.addWidget(isa_text)
        isa_group.setLayout(isa_layout)
        left_panel.addWidget(isa_group)

        main_layout.addLayout(left_panel, 1)

        right_panel = QVBoxLayout()
        
        flags_layout = QHBoxLayout()
        flags_layout.addWidget(QLabel("Flags:"))
        self.led_z = LEDWidget(QColor(255, 50, 50))
        self.led_c = LEDWidget(QColor(50, 150, 255))
        self.led_n = LEDWidget(QColor(255, 200, 0))
        self.led_v = LEDWidget(QColor(200, 0, 255))
        flags_layout.addWidget(QLabel("Z"))
        flags_layout.addWidget(self.led_z)
        flags_layout.addWidget(QLabel("C"))
        flags_layout.addWidget(self.led_c)
        flags_layout.addWidget(QLabel("N"))
        flags_layout.addWidget(self.led_n)
        flags_layout.addWidget(QLabel("V"))
        flags_layout.addWidget(self.led_v)
        flags_layout.addStretch()
        right_panel.addLayout(flags_layout)

        reg_group = QGroupBox("Registers (Space-Time CA History)")
        reg_layout = QGridLayout()
        
        self.reg_widgets = []
        reg_labels = ["PC", "SP", "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
        positions = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1), (4,0), (4,1)]
        
        for i in range(10):
            r, c = positions[i]
            row = QHBoxLayout()
            label = QLabel(f"{reg_labels[i]}:")
            label.setFixedWidth(30)
            row.addWidget(label)
            
            w = CAHistoryWidget(self.cpu.bit_width, height=4, cell_size=6, highlight_active=(i>=2))
            self.reg_widgets.append(w)
            row.addWidget(w)
            container = QWidget()
            container.setLayout(row)
            reg_layout.addWidget(container, r, c)
            
        reg_group.setLayout(reg_layout)
        right_panel.addWidget(reg_group)

        mem_group = QGroupBox("Memory Bank (First 16 Words)")
        mem_layout = QVBoxLayout()
        self.mem_widgets = []
        for i in range(16):
            row = QHBoxLayout()
            label = QLabel(f"M[{i:02d}]:")
            label.setFixedWidth(40)
            row.addWidget(label)
            
            w = CAHistoryWidget(self.cpu.bit_width, height=3, cell_size=4, highlight_active=True)
            self.mem_widgets.append(w)
            row.addWidget(w)
            
            container = QWidget()
            container.setLayout(row)
            mem_layout.addWidget(container)
            
        mem_group.setLayout(mem_layout)
        right_panel.addWidget(mem_group)

        main_layout.addLayout(right_panel, 2)

        self.timer = QTimer()
        self.timer.timeout.connect(self.step_cpu)

    def load_example_program(self, index):
        programs = [
            """
            ; Countdown using CALL/RET and PUSH/POP
            SET R0, 5        
            SET R1, 1        
            
            loop:
            CALL dec_and_save
            CMP R0, R1       
            JNE loop         
            HALT
            
            dec_and_save:
            PUSH R1          
            SUB R0, R1       
            POP R1           
            RET
            """,
            """
            ; Fibonacci Sequence
            SET R0, 1
            SET R1, 1
            SET R2, 6        
            
            loop:
            MOV R3, R1       
            ADD R1, R0       
            MOV R0, R3       
            
            DEC R2
            JNE loop
            HALT
            """,
            """
            ; Visual CA Circular Rotation (ROL)
            SET R0, 1
            
            loop:
            ROL R0
            JMP loop
            """,
            """
            ; Array Sum using Pointers (LOADR)
            SETM 0, 10
            SETM 1, 20
            SETM 2, 30
            SETM 3, 0    

            SET R0, 0    
            SET R1, 0    
            
            loop:
            LOADR R2, R1 
            CMP R2, R3    
            JE end
            ADD R0, R2   
            INC R1       
            JMP loop
            
            end:
            HALT
            """,
            """
            ; Jump Table using JMPR
            SET R0, 2      ; Switch case 2
            SET R1, 0      ; Default case
            
            CMP R0, 1
            JE case1
            CMP R0, 2
            JE case2
            JMP default
            
            case1:
            SET R7, 0x10
            JMPR R7
            case2:
            SET R7, 0x20
            JMPR R7
            default:
            SET R7, 0xFF
            JMPR R7
            HALT
            """,
            """
            ; Block Memory Copy (MEMCPY)
            SETM 0, 0xAA
            SETM 1, 0xBB
            SETM 2, 0xCC
            SETM 3, 0xDD
            
            SET R0, 8     ; Destination
            SET R1, 0     ; Source
            SET R2, 4     ; Length
            MEMCPY R0, R1, R2
            
            HALT
            """
        ]
        self.code_edit.setPlainText(programs[index].strip())
        self.load_program()

    def change_architecture(self):
        idx = self.arch_combo.currentIndex()
        if idx == 0: self.cpu.set_bit_width(16)
        elif idx == 1: self.cpu.set_bit_width(32)
        else: self.cpu.set_bit_width(64)
        
        for w in self.reg_widgets: w.reset_history(self.cpu.bit_width)
        for w in self.mem_widgets: w.reset_history(self.cpu.bit_width)
            
        self.load_program()

    def update_clock_speed(self):
        self.timer.setInterval(self.speed_slider.value())

    def load_program(self):
        code = self.code_edit.toPlainText()
        self.cpu.load_program(code)
        self.log_console.clear()
        self.log_console.appendPlainText(f"[ASM] Program assembled.")
        if self.cpu.labels:
            self.log_console.appendPlainText(f"[ASM] Labels: {self.cpu.labels}")
        self.reset_ca_history()
        self.update_ui()

    def reset_ca_history(self):
        for w in self.reg_widgets: w.reset_history(self.cpu.bit_width)
        for w in self.mem_widgets: w.reset_history(self.cpu.bit_width)

    def step_cpu(self):
        if self.cpu.halted:
            self.timer.stop()
            self.btn_run.setText("Run")
            return

        self.cpu.step()
        self.log_console.appendPlainText(self.cpu.last_instr)
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())
        self.update_ui()

        if self.cpu.halted:
            self.timer.stop()
            self.btn_run.setText("Run")
            self.log_console.appendPlainText("--- CPU HALTED ---")

    def run_cpu(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_run.setText("Run")
        else:
            if self.cpu.halted: self.load_program()
            self.timer.start(self.speed_slider.value())
            self.btn_run.setText("Pause")

    def reset_cpu(self):
        self.timer.stop()
        self.btn_run.setText("Run")
        self.load_program()

    def format_val(self, val):
        if self.base_format == 'hex': return f"0x{val:X}"
        elif self.base_format == 'bin': return f"0b{val:b}"
        return str(val)

    def update_ui(self):
        vals = [self.cpu.pc, self.cpu.sp] + self.cpu.regs
        labels = ["PC", "SP", "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
        
        for i, w in enumerate(self.reg_widgets):
            w.push_state(vals[i], f"{labels[i]}: {self.format_val(vals[i])}")
            if i == self.cpu.active_reg + 2 and self.cpu.active_reg != -1:
                w.flash_active()
        
        for i in range(len(self.mem_widgets)):
            self.mem_widgets[i].push_state(self.cpu.memory[i], "")
            if i == self.cpu.active_mem:
                self.mem_widgets[i].flash_active()
                
        self.led_z.set_on(self.cpu.flags['Z'])
        self.led_c.set_on(self.cpu.flags['C'])
        self.led_n.set_on(self.cpu.flags['N'])
        self.led_v.set_on(self.cpu.flags['V'])
        
        status = "HALTED" if self.cpu.halted else "RUNNING"
        self.setWindowTitle(f"Expanded ISA CA Computer - {self.cpu.bit_width}-bit - {status}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    from PySide6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 33, 38))
    palette.setColor(QPalette.WindowText, QColor(200, 200, 200))
    palette.setColor(QPalette.Base, QColor(22, 24, 28))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, QColor(200, 200, 200))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(45, 50, 55))
    palette.setColor(QPalette.ButtonText, QColor(200, 200, 200))
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())