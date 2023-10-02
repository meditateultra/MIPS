'''
On my honor, I have neither given nor received unauthorized aid on this assignment
'''
import numpy as np
import sys

# 指令类
class instruction:
    category = 0
    op_code = ''
    rs = ''
    rt = ''
    rd = ''
    shift_amount = ''
    func = ''
    ins_str = ''

    def __init__(self,instruction_string):
        category_str = instruction_string[0:2]
        if category_str == '01':
            self.category = 0
        else:
            self.category = 1
        self.op_code = instruction_string[2:6]
        self.rs = instruction_string[6:11]
        self.rt = instruction_string[11:16]
        self.rd = instruction_string[16:21]
        self.shift_amount = instruction_string[21:26]
        self.func = instruction_string[26:]
        self.ins_str = instruction_string

    def __str__(self):
        return self.ins_str

# 读取二进制文件
def read_input_file(filename):
    instructions = []
    with open(filename,"r") as reader:
        for line in reader.readlines():
            instructions.append(line.strip())

    return instructions

# 对寄存器进行格式化输出
def generate_register_name(reg):
    # print(reg)
    reg_name = 'R' + str(int(reg, 2))
    return reg_name

# 补码转换为整数
def complement_to_int(complement):
    res = 0
    if complement[0] == '0':
        res = int(complement,base=2)
    else:
        com_len = len(complement)
        res += -1 * 2**(com_len-1)
        for i in range(1,len(complement)):
            res += int(complement[i]) * 2**(com_len-1-i)
    return res

def int_to_complement(num,width=32):
    return np.binary_repr(num, width=width)

# offset为有符号整数，offset需要左移并且扩充为32位
def sign_extend(complement,width=32):
    res = complement[0] * (width-len(complement)) + complement
    return res

def zero_extend(complement,width=32):
    res = '0' * (width-len(complement)) + complement
    return res

def disassemble_J(instruction):
    immediate = instruction.rs+instruction.rt+instruction.shift_amount+instruction.func + '00'
    ins_arg_str = '#' + str(complement_to_int(immediate))
    return ins_arg_str

def simulate_J(instruction,PC,data_address,regsters,memory):
    immediate = instruction.rs + instruction.rt + instruction.shift_amount + instruction.func + '00'
    PC[0] = complement_to_int(immediate)


def disassemble_JR(instruction):
    ins_arg_str = generate_register_name(instruction.rs)
    return ins_arg_str

def simulate_JR(instruction,PC,data_address,regsters,memory):
    PC[0] = registers[complement_to_int(instruction.rs)]


def disassemble_BEQ(instruction):
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    # 右移2位数
    offset = instruction.rd + instruction.shift_amount + instruction.func + '00'
    # 符号扩充为32位
    offset = sign_extend(offset,width=32)

    ins_arg_str = rs + ', ' + rt + ', ' + '#' + str(complement_to_int(offset))

    return ins_arg_str


def simulate_BEQ(instruction,PC,data_address,regsters,memory):
    # 右移2位数
    offset = instruction.rd + instruction.shift_amount + instruction.func + '00'
    # 符号扩充为32位
    offset = sign_extend(offset, width=32)

    if registers[int(instruction.rs, 2)] == registers[int(instruction.rt, 2)]:
        PC[0] += complement_to_int(offset) + 4
    else:
        PC[0] += 4

# BLTZ rs, offset
def disassemble_BLTZ(instruction):
    rs = generate_register_name(instruction.rs)
    offset = instruction.rd + instruction.rt + instruction.shift_amount + instruction.func + '00'
    # 符号扩充为32位
    offset = sign_extend(offset, width=32)

    ins_arg_str = rs + ', ' + '#' + str(complement_to_int(offset))

    return ins_arg_str

def simulate_BLTZ(instruction,PC,data_address,regsters,memory):
    offset = instruction.rd + instruction.rt + instruction.shift_amount + instruction.func + '00'
    # 符号扩充为32位
    offset = sign_extend(offset, width=32)

    if registers[int(instruction.rs, 2)] < 0:
        PC[0] += complement_to_int(offset) + 4
    else:
        PC[0] += 4

# BGTZ rs, offset
def disassemble_BGTZ(instruction):
    rs = generate_register_name(instruction.rs)
    offset = instruction.rd + instruction.rt + instruction.shift_amount + instruction.func + '00'
    # 符号扩充为32位
    offset = sign_extend(offset, width=32)

    ins_arg_str = rs + ', ' + '#' + str(complement_to_int(offset))

    return ins_arg_str

def simulate_BGTZ(instruction,PC,data_address,regsters,memory):
    offset = instruction.rd + instruction.rt + instruction.shift_amount + instruction.func + '00'
    # 符号扩充为32位
    offset = sign_extend(offset, width=32)

    if registers[int(instruction.rs, 2)] > 0:
        PC[0] += complement_to_int(offset) + 4
    else:
        PC[0] += 4

# BREAK指令无输出
def disassemble_BREAK(instruction):
    return ""

def simulate_BREAK(instruction,PC,data_address,regsters,memory):
    PC[0] += 4

# SW rt, offset(base)
def disassemble_SW(instruction):
    base = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    offset = instruction.rd + instruction.shift_amount + instruction.func
    ins_arg_str = rt + ', ' + str(complement_to_int(offset)) + '(' + base + ')'
    return ins_arg_str

def simulate_SW(instruction,PC,data_address,regsters,memory):
    base = int(instruction.rs,2)

    offset = instruction.rd + instruction.shift_amount + instruction.func

    valid_address = int((registers[base] + complement_to_int(offset) - data_address) / 4)
    memory[int(valid_address)] = registers[int(instruction.rt, 2)]

    PC[0] += 4


# LW rt, offset(base)
def disassemble_LW(instruction):
    base = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    offset = instruction.rd + instruction.shift_amount + instruction.func
    ins_arg_str = rt + ', ' + str(complement_to_int(offset)) + '(' + base + ')'
    return ins_arg_str

def simulate_LW(instruction,PC,data_address,regsters,memory):
    base = int(instruction.rs,2)

    offset = instruction.rd + instruction.shift_amount + instruction.func

    valid_address = int((registers[base] + complement_to_int(offset) - data_address) / 4)
    registers[int(instruction.rt, 2)] = memory[int(valid_address)]

    PC[0] += 4


# SLL rd, rt, sa
def disassemble_SLL(instruction):
    rd = generate_register_name(instruction.rd)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rt + ', ' + '#' + str(complement_to_int(instruction.shift_amount))
    return ins_arg_str

def simulate_SLL(instruction,PC,data_address,regsters,memory):
    registers[int(instruction.rd, 2)] = (registers[int(instruction.rt, 2)] << int(instruction.shift_amount, 2)) & (2 ** 32 - 1)

    PC[0] += 4

# SRL rd, rt, sa
def disassemble_SRL(instruction):
    rd = generate_register_name(instruction.rd)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rt + ', ' + '#' + str(complement_to_int(instruction.shift_amount))
    return ins_arg_str

def simulate_SRL(instruction,PC,data_address,regsters,memory):
    sa = complement_to_int(instruction.shift_amount)

    s = 32 - sa
    res = int_to_complement(registers[int(instruction.rt, 2)])[:s]

    res = '0' * sa + res

    registers[int(instruction.rd, 2)] = complement_to_int(res)

    PC[0] += 4

# SRA rd, rt, sa
def disassemble_SRA(instruction):
    rd = generate_register_name(instruction.rd)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rt + ', ' + '#' + str(complement_to_int(instruction.shift_amount))
    return ins_arg_str

def simulate_SRA(instruction,PC,data_address,regsters,memory):
    sa = complement_to_int(instruction.shift_amount)

    s = 32 - sa
    res = int_to_complement(registers[int(instruction.rt, 2)])[:s]

    res = res[0] * sa + res

    registers[int(instruction.rd, 2)] = complement_to_int(res)
    PC[0] += 4

# NOP
def disassemble_NOP(instruction):
    return ""

def simulate_NOP(instruction,PC,data_address,regsters,memory):
    PC[0] += 4

# ADD rd, rs, rt
def disassemble_ADD(instruction):
    # print(instruction.ins_str)
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str

def simulate_ADD(instruction,PC,data_address,regsters,memory):
    registers[int(instruction.rd, 2)] = registers[int(instruction.rs, 2)] + registers[int(instruction.rt, 2)]
    PC[0] += 4

# SUB rd, rs, rt
def disassemble_SUB(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str


def simulate_SUB(instruction,PC,data_address,regsters,memory):
    registers[int(instruction.rd, 2)] = registers[int(instruction.rs, 2)] - registers[int(instruction.rt, 2)]
    PC[0] += 4

def disassemble_MUL(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str

def simulate_MUL(instruction,PC,data_address,regsters,memory):
    registers[int(instruction.rd, 2)] = registers[int(instruction.rs, 2)] * registers[int(instruction.rt, 2)]
    PC[0] += 4

def disassemble_AND(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str

def simulate_AND(instruction,PC,data_address,regsters,memory):
    registers[int(instruction.rd, 2)] = registers[int(instruction.rs, 2)] & registers[int(instruction.rt, 2)]
    PC[0] += 4

def disassemble_OR(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str


def simulate_OR(instruction,PC,data_address,regsters,memory):
    registers[int(instruction.rd, 2)] = registers[int(instruction.rs, 2)] | registers[int(instruction.rt, 2)]
    PC[0] += 4

def disassemble_XOR(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str


def simulate_XOR(instruction,PC,data_address,regsters,memory):
    registers[int(instruction.rd, 2)] = registers[int(instruction.rs, 2)] ^ registers[int(instruction.rt, 2)]
    PC[0] += 4

def disassemble_NOR(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str

def simulate_NOR(instruction,PC,data_address,regsters,memory):
    registers[int(instruction.rd, 2)] = ~(registers[int(instruction.rs, 2)]) | registers[int(instruction.rt, 2)]
    PC[0] += 4

def disassemble_SLT(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str

def simulate_SLT(instruction,PC,data_address,regsters,memory):
    if registers[int(instruction.rs, 2)] < registers[int(instruction.rt, 2)]:
        registers[int(instruction.rd, 2)] = 1
    else:
        registers[int(instruction.rd, 2)] = 0

    PC[0] += 4

# ADDI rt, rs, immediate
def disassemble_ADDI(instruction):
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)
    immediate = instruction.rd + instruction.shift_amount + instruction.func
    immediate = sign_extend(immediate)
    ins_arg_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))

    return ins_arg_str


def simulate_ADDI(instruction,PC,data_address,regsters,memory):
    immediate = instruction.rd + instruction.shift_amount + instruction.func
    immediate = sign_extend(immediate)
    registers[int(instruction.rt, 2)] = registers[int(instruction.rs, 2)] + complement_to_int(immediate)
    PC[0] += 4

def disassemble_ANDI(instruction):
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)
    immediate = instruction.rd + instruction.shift_amount + instruction.func
    ins_arg_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))

    return ins_arg_str

def simulate_ANDI(instruction,PC,data_address,regsters,memory):
    immediate = instruction.rd + instruction.shift_amount + instruction.func
    immediate = zero_extend(immediate)
    registers[int(instruction.rt, 2)] = registers[int(instruction.rs, 2)] & complement_to_int(immediate)
    PC[0] += 4

def disassemble_ORI(instruction):
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)
    immediate = instruction.rd + instruction.shift_amount + instruction.func


    ins_arg_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))

    return ins_arg_str

def simulate_ORI(instruction,PC,data_address,regsters,memory):
    immediate = instruction.rd + instruction.shift_amount + instruction.func
    immediate = zero_extend(immediate)
    registers[int(instruction.rt, 2)] = registers[int(instruction.rs, 2)] | complement_to_int(immediate)
    PC[0] += 4

def disassemble_XORI(instruction):
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)
    immediate = instruction.rd + instruction.shift_amount + instruction.func

    ins_arg_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))

    return ins_arg_str

def simulate_XORI(instruction,PC,data_address,regsters,memory):
    immediate = instruction.rd + instruction.shift_amount + instruction.func
    immediate = zero_extend(immediate)
    registers[int(instruction.rt, 2)] = registers[int(instruction.rs, 2)] ^ complement_to_int(immediate)
    PC[0] += 4

op_code_dict = {
    '0000':['J','ADD'],
    '0001':['JR','SUB'],
    '0010':['BEQ','MUL'],
    '0011':['BLTZ','AND'],
    '0100':['BGTZ','OR'],
    '0101':['BREAK','XOR'],
    '0110':['SW','NOR'],
    '0111':['LW','SLT'],
    '1000':['SLL','ADDI'],
    '1001':['SRL','ANDI'],
    '1010':['SRA','ORI'],
    '1011':['NOP','XORI']
}

function_dict = {
    'J':['disassemble_J','simulate_J'],
    'JR':['disassemble_JR','simulate_JR'],
    'BEQ':['disassemble_BEQ','simulate_BEQ'],
    'BLTZ':['disassemble_BLTZ','simulate_BLTZ'],
    'BGTZ':['disassemble_BGTZ','simulate_BGTZ'],
    'BREAK':['disassemble_BREAK','simulate_BREAK'],
    'SW':['disassemble_SW','simulate_SW'],
    'LW':['disassemble_LW','simulate_LW'],
    'SLL':['disassemble_SLL','simulate_SLL'],
    'SRL':['disassemble_SRL','simulate_SRL'],
    'SRA':['disassemble_SRA','simulate_SRA'],
    'NOP':['disassemble_NOP','simulate_NOP'],
    'ADD':['disassemble_ADD','simulate_ADD'],
    'SUB':['disassemble_SUB','simulate_SUB'],
    'MUL':['disassemble_MUL','simulate_MUL'],
    'AND':['disassemble_AND','simulate_AND'],
    'OR':['disassemble_OR','simulate_OR'],
    'XOR':['disassemble_XOR','simulate_XOR'],
    'NOR':['disassemble_NOR','simulate_NOR'],
    'SLT':['disassemble_SLT','simulate_SLT'],
    'ADDI':['disassemble_ADDI','simulate_ADDI'],
    'ANDI':['disassemble_ANDI','simulate_ANDI'],
    'ORI':['disassemble_ORI','simulate_ORI'],
    'XORI':['disassemble_XORI','simulate_XORI'],
}


def generate_disassembly(instruction,PC):
    ins_name_str = op_code_dict[instruction.op_code][instruction.category]
    ins_args_str = eval(function_dict[ins_name_str][0])(instruction)

    # if ins_name_str != 'BREAK':
    #     disassemble_file.write(str(instruction) + '\t' + str(PC[0]) + '\t' + ins_name_str + ' ' + ins_args_str + '\n')
    # else:
    #     disassemble_file.write(str(instruction) + '\t' + str(PC[0]) + '\t' + ins_name_str + '\n')
    #     return True
    if ins_name_str == 'BREAK':
        return True

    return False

# registers
registers = [0] * 32

class ScoreBoarding:
    def __init__(self, instructions, PC, data_address, registers, memory, simulation_file):
        self.instructions = instructions
        self.PC = PC
        self.data_address = data_address
        self.registers = registers
        self.memory = memory

        # 输出文件
        self.simulation_file = simulation_file

        # scoreboarding单元
        self.pre_issue = []
        self.pre_alu1 = []
        self.pre_alu2 = []
        self.pre_mem = []
        self.post_alu2 = []
        self.post_mem = []
        self.register_status = [True] * 32

        # 判断是否stall
        self.is_stall = False
        # 判断是否读取到break指令
        self.is_break = False

        # 当前cycle发生stall的指令
        self.stalled_instruction = None
        # 当前周期execute的指令
        # 只有branch指令
        self.executed_instruction = None

        # 用于判断pre_issue中的data_hazard
        self.register_write = [4] * 32
        self.register_read = [4] * 32

        # 判断之前的store是否完成
        # store必须按序
        self.store_ready = True

    # 处理branch指令
    def branch(self, instruction):
        self.is_stall = False
        # 处理J, JR, BEQ, BLTZ, BGTZ指令

        ins_name_str = op_code_dict[instruction.op_code][instruction.category]
        if ins_name_str == 'J':
            immediate = instruction.rs + instruction.rt + instruction.shift_amount + instruction.func + '00'
            self.PC[0] = complement_to_int(immediate)
        elif ins_name_str == 'JR':
            self.PC[0] = registers[complement_to_int(instruction.rs)]
        elif ins_name_str == 'BEQ':
            rs = int(instruction.rs, 2)
            rt = int(instruction.rt, 2)

            # rs,rt 是否可用
            if not self.register_status[rs] or not self.register_status[rt]:
                self.is_stall = True
            else:
                if self.registers[rs] == self.registers[rt]:
                    # 右移2位数
                    offset = instruction.rd + instruction.shift_amount + instruction.func + '00'
                    # 符号扩充为32位
                    offset = sign_extend(offset, width=32)
                    self.PC[0] += complement_to_int(offset) + 4
                else:
                    self.PC[0] += 4
        elif ins_name_str == 'BLTZ':
            rs = int(instruction.rs,2)
            if not self.register_status[rs]:
                self.is_stall = True
            else:
                if self.registers[rs] < 0:
                    offset = instruction.rd + instruction.shift_amount + instruction.func + '00'
                    # 符号扩充为32位
                    offset = sign_extend(offset, width=32)
                    self.PC[0] += complement_to_int(offset) + 4
                else:
                    self.PC[0] += 4

        elif ins_name_str == 'BGTZ':
            rs = int(instruction.rs, 2)
            if not self.register_status[rs]:
                self.is_stall = True
            else:
                if self.registers[rs] > 0:
                    offset = instruction.rd + instruction.shift_amount + instruction.func + '00'
                    # 符号扩充为32位
                    offset = sign_extend(offset, width=32)
                    self.PC[0] += complement_to_int(offset) + 4
                else:
                    self.PC[0] += 4

        return not self.is_stall

    # 读取指令后，需要对目的寄存器上锁
    # 不用于判断data_hazard
    def lock_register(self,instruction):
        rd = int(instruction.rd,2)
        rt = int(instruction.rt,2)

        ins_name_str = op_code_dict[instruction.op_code][instruction.category]
        if ins_name_str in ['ADDI', 'ANDI', 'ORI', 'XORI','LW']:
            self.register_status[rt] = False
        elif ins_name_str in ['SLL', 'SRL', 'SRA','ADD', 'SUB', 'MUL', 'AND', 'OR', 'XOR', 'NOR', 'SLT']:
            self.register_status[rd] = False

    # 取指令单元
    def if_unit(self):
        instructions_fetched = []
        # 取指令之前Executed置空
        self.executed_instruction = None
        # 上一个cycle出现stall,查看此时能否可以跳转
        if self.is_stall:
            temp_ins = self.stalled_instruction
            if self.branch(temp_ins):
                # Executed需要记录branch指令
                self.executed_instruction = temp_ins
                self.stalled_instruction = None
        # 上一个cycle未出现stall
        else:
            num_fetched = 0
            while not self.is_break and num_fetched < 2 and len(self.pre_issue) + num_fetched < 4:
                ins_index = int((self.PC[0]-256)/4)
                temp_ins = instructions[ins_index]
                ins_name_str = op_code_dict[temp_ins.op_code][temp_ins.category]
                if ins_name_str == 'BREAK':
                    # Executed需要记录break指令
                    self.executed_instruction = temp_ins
                    self.is_break = True
                else:
                    if ins_name_str in ['J','JR','BEQ','BLTZ','BGTZ']:
                        if not self.branch(temp_ins):
                            self.stalled_instruction = temp_ins
                            self.is_stall = True
                        else:
                            # Executed需要记录break指令
                            self.executed_instruction = temp_ins
                        break
                    else:
                        instructions_fetched.append(temp_ins)
                        self.lock_register(temp_ins)
                        self.PC[0] += 4
                        num_fetched += 1
        return instructions_fetched

    # 只需要检查pre_issue中的即可
    # register_write存当前写该寄存器的指令在pre_issue中的index
    # register_read存当前读该寄存器的指令在pre_issue中的index
    def is_issue_ready(self,instruction, index):
        # 是否有data_hazard
        ready = False
        previous_store = self.store_ready

        ins_name_str = op_code_dict[instruction.op_code][instruction.category]

        # 涉及rs,rt,rd
        if ins_name_str in ['ADD', 'SUB', 'MUL', 'AND', 'OR', 'XOR', 'NOR', 'SLT']:
            rs = int(instruction.rs,2)
            rd = int(instruction.rd,2)
            rt = int(instruction.rt,2)

            # 在pre_issue中判断是否有data_hazard
            if self.register_write[rs] >= index and self.register_write[rt] >= index and self.register_read[rd] >= index and self.register_write[rd] >= index:
                ready = True

            # 更新index
            if self.register_write[rd] > index:
                self.register_write[rd] = index
            if self.register_read[rs] > index:
                self.register_read[rs] = index
            if self.register_read[rt] > index:
                self.register_read[rt] = index
        # 涉及rt,rs
        elif ins_name_str in ['ADDI','ANDI','ORI','XORI']:
            rs = int(instruction.rs, 2)
            rt = int(instruction.rt, 2)

            if self.register_write[rs] >= index and self.register_read[rt] >= index and self.register_write[rt] >= index:
                ready = True

            if self.register_read[rs] > index:
                self.register_read[rs] = index
            if self.register_write[rt] > index:
                self.register_write[rt] = index
        # 涉及rd,rt
        elif ins_name_str in ['SLL','SRL','SRA']:
            rt = int(instruction.rt, 2)
            rd = int(instruction.rd, 2)

            if self.register_write[rt] >= index and self.register_read[rd] >= index and self.register_write[rd] >= index:
                ready = True

            if self.register_read[rt] > index:
                self.register_read[rt] = index
            if self.register_write[rd] > index:
                self.register_write[rd] = index
        # 涉及mem
        elif ins_name_str == 'LW':
            rt = int(instruction.rt, 2)
            base = int(instruction.rs, 2)

            if self.register_status[base] and self.register_status[rt] and self.store_ready:
                ready = True
                self.register_status[rt] = False

            if self.register_write[base] >= index and self.register_read[rt] >= index and self.register_write[rt] >= index and self.store_ready:
                ready = True

            if self.register_write[rt] > index:
                self.register_write[rt] = index
            if self.register_read[base] > index:
                self.register_read[base] = index
        elif ins_name_str == 'SW':
            rt = int(instruction.rt, 2)
            base = int(instruction.rs, 2)

            if self.register_write[rt] >= index and self.register_write[base] >= index and self.store_ready:
                ready = True
            else:
                previous_store = False
            if self.register_read[rt] > index:
                self.register_read[rt] = index
            if self.register_read[base] > index:
                self.register_read[base] = index

        return ready, previous_store

    # 发射单元，读取is_issue_ready为True的指令
    def issue_unit(self):
        alu1 = []
        alu2 = []
        issued_alu1 = 0
        issued_alu2 = 0

        # 一个cycle可以更新一次
        self.store_ready = True
        i = 0
        while i < len(self.pre_issue):
            if (issued_alu1 == 1 and issued_alu2 == 1) or len(self.pre_alu1) == 2 or len(self.pre_alu2) == 2:
                break

            ins = self.pre_issue[i]
            ins_name_str = op_code_dict[ins.op_code][ins.category]

            if ins_name_str in ['LW','SW']:
                if issued_alu1 == 1:
                    break
            else:
                if issued_alu2 == 1:
                    break
            # 判断当前指令是否可以issue
            # previous_store判断之前的store指令是否完成
            ready, previous_store = self.is_issue_ready(ins,i)

            self.store_ready = previous_store and self.store_ready

            if ready:
                rs = int(ins.rs,2)
                rt = int(ins.rt, 2)
                rd = int(ins.rd, 2)

                if ins_name_str in ['LW', 'SW']:
                    if ins_name_str == "LW":
                        # 更新index
                        if self.register_read[rs] > i:
                            self.register_read[rs] = 4
                        self.register_write[rt] = -1
                    elif ins_name_str == "SW":
                        if self.register_read[rt] >= i:
                            self.register_read[rt] = 4
                        if self.register_read[rs] >= i:
                            self.register_read[rs] = 4
                    issued_alu1 += 1
                    alu1.append(ins)
                    self.pre_issue.remove(ins)
                else:
                    if issued_alu2 == 1:
                        break
                    if ins_name_str in ["ADD", "SUB", "MUL", "AND", "OR", "XOR", "NOR"]:
                        if self.register_read[rt] >= i:
                            self.register_read[rt] = 4
                        if self.register_read[rs] >= i:
                            self.register_read[rs] = 4
                        self.register_write[rd] = -1
                    if ins_name_str in ["ADDI", "ANDI", "ORI", "XORI"]:
                        if self.register_read[rs] >= i:
                            self.register_read[rs] = 4
                        self.register_write[rt] = -1
                    if ins_name_str in ['SLL', 'SRL', 'SRA']:
                        if self.register_read[rt] >= i:
                            self.register_read[rt] = 4
                        self.register_write[rd] = -1
                    issued_alu2 += 1
                    alu2.append(ins)
                    self.pre_issue.remove(ins)
            i += 1

        return alu1,alu2

    # alu1单元
    def alu1(self):
        pre_mem = []
        if len(self.pre_alu1) > 0:
            ins = self.pre_alu1.pop(0)
            pre_mem.append(ins)
        return pre_mem

    # alu2单元
    def alu2(self):
        post_alu2 = []
        if len(self.pre_alu2) > 0:
            ins = self.pre_alu2.pop(0)
            post_alu2.append(ins)
        return post_alu2

    # mem单元
    def mem(self):
        post_mem = []
        if len(self.pre_mem) == 1:
            ins = self.pre_mem.pop(0)
            ins_name_str = op_code_dict[ins.op_code][ins.category]

            if ins_name_str == 'SW':
                eval(function_dict[ins_name_str][1])(ins, self.PC, self.data_address, self.registers, self.memory)
                self.PC[0] -= 4
            else:
                post_mem.append(ins)
        return post_mem

    # wb单元
    def wb(self):
        # 可以同时wb两个
        if len(self.post_mem) == 1:
            ins = self.post_mem.pop(0)
            ins_name_str = op_code_dict[ins.op_code][ins.category]
            eval(function_dict[ins_name_str][1])(ins, self.PC, self.data_address, self.registers, self.memory)
            self.PC[0] -= 4
            rt = int(ins.rt,2)
            # 更新寄存器状态
            self.register_status[rt] = True
            self.register_write[rt] = 4

        if len(self.post_alu2) == 1:
            ins = self.post_alu2.pop(0)
            ins_name_str = op_code_dict[ins.op_code][ins.category]
            eval(function_dict[ins_name_str][1])(ins, self.PC, self.data_address, self.registers, self.memory)
            self.PC[0] -= 4

            rd = int(ins.rd,2)
            rt = int(ins.rt,2)

            # wb后更新寄存器状态
            if ins_name_str in ['SLL', 'SRA', 'SRL', 'ADD', 'SUB', 'AND', 'OR', 'XOR', 'NOR', 'SLT']:
                self.register_status[rd] = True
                self.register_write[rd] = 4
            if ins_name_str in ['ADDI', 'ANDI', 'ORI', 'XORI']:
                self.register_status[rt] = True
                self.register_write[rt] = 4

    # 输出函数
    def output(self,cycle):
        output_str = '--------------------\n'
        output_str += 'Cycle:' + str(cycle) + '\n'
        output_str += '\n'

        # IF Unit
        output_str += 'IF Unit:\n'

        # Waiting Instruction
        output_str += '\tWaiting Instruction:'
        if self.stalled_instruction:
            ins = self.stalled_instruction
            ins_name_str = op_code_dict[ins.op_code][ins.category]
            ins_args_str = eval(function_dict[ins_name_str][0])(ins)

            output_str += ' [' + ins_name_str + ' ' + ins_args_str + ']\n'
        else:
            output_str += '\n'

        # Executed Instruction
        output_str += '\tExecuted Instruction:'
        if self.executed_instruction:
            ins = self.executed_instruction
            ins_name_str = op_code_dict[ins.op_code][ins.category]
            ins_args_str = eval(function_dict[ins_name_str][0])(ins)

            output_str += ' [' + ins_name_str + ' ' + ins_args_str + ']\n'
        else:
            output_str += '\n'

        # Pre-Issue Queue
        output_str += 'Pre-Issue Queue:\n'
        for i in range(4):
            temp = ''
            if i < len(self.pre_issue):
                temp_ins = self.pre_issue[i]
                ins_name_str = op_code_dict[temp_ins.op_code][temp_ins.category]
                # print(ins_name_str)
                ins_args_str = eval(function_dict[ins_name_str][0])(temp_ins)
                temp = ' [' + ins_name_str + ' ' + ins_args_str + ']'
            output_str += '\tEntry ' + str(i) + ':' + temp + '\n'

        # Pre-ALU1 Queue
        output_str += 'Pre-ALU1 Queue:\n'
        for i in range(2):
            temp = ''
            if i < len(self.pre_alu1):
                temp_ins = self.pre_alu1[i]
                ins_name_str = op_code_dict[temp_ins.op_code][temp_ins.category]
                ins_args_str = eval(function_dict[ins_name_str][0])(temp_ins)
                temp = ' [' + ins_name_str + ' ' + ins_args_str + ']'
            output_str += '\tEntry ' + str(i) + ':' + temp + '\n'

        # Pre-MEM Queue
        output_str += 'Pre-MEM Queue:'
        if len(self.pre_mem) > 0:
            temp_ins = self.pre_mem[0]
            ins_name_str = op_code_dict[temp_ins.op_code][temp_ins.category]
            ins_args_str = eval(function_dict[ins_name_str][0])(temp_ins)
            output_str += ' [' + ins_name_str + ' ' + ins_args_str + ']'
        output_str += '\n'


        # Post-MEM Queue
        output_str += 'Post-MEM Queue:'
        if len(self.post_mem) > 0:
            temp_ins = self.post_mem[0]
            ins_name_str = op_code_dict[temp_ins.op_code][temp_ins.category]
            ins_args_str = eval(function_dict[ins_name_str][0])(temp_ins)
            output_str += ' [' + ins_name_str + ' ' + ins_args_str + ']'
        output_str += '\n'

        # Pre-ALU2 Queue
        output_str += 'Pre-ALU2 Queue:\n'
        for i in range(2):
            temp = ''
            if i < len(self.pre_alu2):
                temp_ins = self.pre_alu2[i]
                ins_name_str = op_code_dict[temp_ins.op_code][temp_ins.category]
                ins_args_str = eval(function_dict[ins_name_str][0])(temp_ins)
                temp = ' [' + ins_name_str + ' ' + ins_args_str + ']'
            output_str += '\tEntry ' + str(i) + ':' + temp + '\n'

        # Post-ALU2 Queue
        output_str += 'Post-ALU2 Queue:'
        if len(self.post_alu2) > 0:
            temp_ins = self.post_alu2[0]
            ins_name_str = op_code_dict[temp_ins.op_code][temp_ins.category]
            ins_args_str = eval(function_dict[ins_name_str][0])(temp_ins)
            output_str += ' [' + ins_name_str + ' ' + ins_args_str + ']'
        output_str += '\n'

        # registers
        output_str += '\nRegisters' + '\n'
        output_str += 'R00:' + '\t' + str(self.registers[0]) + '\t' + str(self.registers[1]) + '\t' + str(self.registers[2]) + '\t' + str(self.registers[3]) + '\t' + str(self.registers[4]) + '\t' + str(self.registers[5]) + '\t' + str(self.registers[6]) + '\t' + str(self.registers[7]) + '\n'
        output_str += 'R08:' + '\t' + str(self.registers[8]) + '\t' + str(self.registers[9]) + '\t' + str(self.registers[10]) + '\t' + str(self.registers[11]) + '\t' + str(self.registers[12]) + '\t' + str(self.registers[13]) + '\t' + str(self.registers[14]) + '\t' + str(self.registers[15]) + '\n'
        output_str += 'R16:' + '\t' + str(self.registers[16]) + '\t' + str(self.registers[17]) + '\t' + str(self.registers[18]) + '\t' + str(self.registers[19]) + '\t' + str(self.registers[20]) + '\t' + str(self.registers[21]) + '\t' + str(self.registers[22]) + '\t' + str(self.registers[23]) + '\n'
        output_str += 'R24:' + '\t' + str(self.registers[24]) + '\t' + str(self.registers[25]) + '\t' + str(self.registers[26]) + '\t' + str(self.registers[27]) + '\t' + str(self.registers[28]) + '\t' + str(self.registers[29]) + '\t' + str(self.registers[30]) + '\t' + str(self.registers[31]) + '\n'
        output_str += '\n'

        # data segement
        output_str += 'Data' + '\n'

        row = 0
        col = 0

        for mem in self.memory:
            if col == 0:
                output_str += str(self.data_address + row * 32) + ':\t'
            col += 1
            if col == 8:
                output_str += str(mem) + '\n'
                col = 0
                row += 1
            else:
                output_str +=  str(mem) + '\t'

        # 不足的补0
        while col > 0 and col < 8:
            col += 1
            if col == 8:
                output_str += '0\n'
            else:
                output_str += '0\t'

        return output_str


    def simulate(self):
        output_str = ''
        cycle = 1
        while True:
            instructions_fetched = self.if_unit()
            alu1,alu2 = self.issue_unit()
            pre_mem = self.alu1()
            post_mem = self.mem()
            post_alu2 = self.alu2()
            self.wb()

            self.pre_issue.extend(instructions_fetched)
            self.pre_alu1.extend(alu1)
            self.pre_alu2.extend(alu2)
            self.pre_mem.extend(pre_mem)
            self.post_alu2.extend(post_alu2)
            self.post_mem.extend(post_mem)

            output_str += self.output(cycle)
            cycle+=1

            if self.is_break:
                break
        self.simulation_file.write(output_str)

if __name__ ==  '__main__':

    # 以命令行方式读取
    # input_filename = sys.argv[1]
    input_filename = 'sample.txt'

    ins_strs = read_input_file(input_filename)
    instructions = []
    index = 0
    for ins in ins_strs:
        instructions.append(instruction(ins))

    PC = [256]
    # disassemble_file = open('disassembly.txt','w')
    data_index = 0

    memory = []

    for ins in instructions:
        flag = generate_disassembly(ins,PC)
        PC[0] += 4
        data_index += 1
        if flag:
            break

    # data段
    data_ins = instructions[data_index:]

    count = 0
    for data in data_ins:
        current_data_index = int((PC[0]-256-4*data_index)/4)
        memory.append(complement_to_int(str(data)))
        # disassemble_file.write(str(data) + '\t' + str(PC[0]) + '\t' + str(complement_to_int(str(data))) + '\n')
        PC[0] += 4
        count += 1

    # disassemble_file.close()

    simulation_file = open('simulation.txt','w')

    PC[0] = 256

    data_address = data_index*4 + 256

    scoreboarding = ScoreBoarding(instructions, PC, data_address, registers, memory, simulation_file)
    scoreboarding.simulate()

    simulation_file.close()









