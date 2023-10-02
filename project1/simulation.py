from utils import *

def simulation_J(instruction, PC, data_address, registers, memory):
    immediate = instruction.rs + instruction.rt + instruction.rd + \
        instruction.shift_amount + instruction.func + '00'
    PC[0] = complement_to_int(immediate)



def simulation_JR(instruction, PC, data_address, registers, memory):
    PC[0] = registers[complement_to_int(instruction.rs)]


def simulation_BEQ(instruction, PC, data_address, registers, memory):
    offset = instruction.rd + instruction.shift_amount + instruction.func + '00'
    offset = sign_extend(offset, width=32)
    if registers[int(instruction.rs, 2)] == registers[int(instruction.rt, 2)]:
        PC[0] += complement_to_int(offset) + 4
    else:
        PC[0] += 4


def simulation_BLTZ(instruction, PC, data_address, registers, memory):
    offset = instruction.rt + instruction.rd + \
        instruction.shift_amount + instruction.func + '00'
    offset = sign_extend(offset, width=32)
    if registers[int(instruction.rs, 2)] < 0:
        PC[0] += complement_to_int(offset) + 4
    else:
        PC[0] += 4


def simulation_BGTZ(instruction, PC, data_address, registers, memory):
    offset = instruction.rt + instruction.rd + \
        instruction.shift_amount + instruction.func + '00'
    offset = sign_extend(offset, width=32)
    if registers[int(instruction.rs, 2)] > 0:
        PC[0] += complement_to_int(offset) + 4
    else:
        PC[0] += 4


def simulation_BREAK(instruction, PC, data_address, registers, memory):
    PC[0] += 4


def simulation_SW(instruction, PC, data_address, registers, memory):
    base = int(instruction.rs, 2)
    offset = instruction.rd + instruction.shift_amount + instruction.func
    valid_address = int(
        (registers[base] +
         complement_to_int(offset) -
         data_address) /
        4)
    memory[int(valid_address)] = registers[int(instruction.rt, 2)]
    PC[0] += 4


def simulation_LW(instruction, PC, data_address, registers, memory):
    base = int(instruction.rs, 2)
    offset = instruction.rd + instruction.shift_amount + instruction.func
    valid_address = int(
        (registers[base] +
         complement_to_int(offset) -
         data_address) / 4)
    registers[int(instruction.rt, 2)] = memory[int(valid_address)]
    PC[0] += 4


def simulation_SLL(instruction, PC, data_address, registers, memory):
    registers[int(instruction.rd, 2)] = (registers[int(instruction.rt, 2)] << int(
        instruction.shift_amount, 2)) & (2 ** 32 - 1)
    PC[0] += 4


def simulation_SRL(instruction, PC, data_address, registers, memory):
    sa = complement_to_int(instruction.shift_amount)
    s = 32 - sa
    res = int_to_complement(registers[int(instruction.rt, 2)])[:s]
    res = '0' * sa + res
    registers[int(instruction.rd, 2)] = complement_to_int(res)
    PC[0] += 4


def simulation_SRA(instruction, PC, data_address, registers, memory):
    sa = complement_to_int(instruction.shift_amount)
    s = 32 - sa
    res = int_to_complement(registers[int(instruction.rt, 2)])[:s]
    res = res[0] * sa + res
    registers[int(instruction.rd, 2)] = complement_to_int(res)
    PC[0] += 4


def simulation_NOP(instruction, PC, data_address, registers, memory):
    PC[0] += 4


def simulation_ADD(instruction, PC, data_address, registers, memory):
    if instruction.category == 0:
        registers[int(instruction.rd, 2)] = registers[int(
            instruction.rs, 2)] + registers[int(instruction.rt, 2)]
        PC[0] += 4
    else:
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        registers[int(instruction.rt, 2)] = registers[int(
            instruction.rs, 2)] + complement_to_int(immediate)
        PC[0] += 4


def simulation_SUB(instruction, PC, data_address, registers, memory):
    if instruction.category == 0:
        registers[int(instruction.rd, 2)] = registers[int(
            instruction.rs, 2)] - registers[int(instruction.rt, 2)]
        PC[0] += 4
    else:
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        registers[int(instruction.rt, 2)] = registers[int(
            instruction.rs, 2)] - complement_to_int(immediate)
        PC[0] += 4


def simulation_MUL(instruction, PC, data_address, registers, memory):
    if instruction.category == 0:
        registers[int(instruction.rd, 2)] = registers[int(
            instruction.rs, 2)] * registers[int(instruction.rt, 2)]
        PC[0] += 4
    else:
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        registers[int(instruction.rt, 2)] = registers[int(
            instruction.rs, 2)] * complement_to_int(immediate)
        PC[0] += 4


def simulation_AND(instruction, PC, data_address, registers, memory):
    if instruction.category == 0:
        registers[int(instruction.rd, 2)] = registers[int(
            instruction.rs, 2)] & registers[int(instruction.rt, 2)]
        PC[0] += 4
    else:
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        registers[int(instruction.rt, 2)] = registers[int(
            instruction.rs, 2)] & complement_to_int(immediate)
        PC[0] += 4


def simulation_NOR(instruction, PC, data_address, registers, memory):
    if instruction.category == 0:
        registers[int(instruction.rd, 2)] = ~(
            registers[int(instruction.rs, 2)]) | registers[int(instruction.rt, 2)]
        PC[0] += 4
    else:
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        registers[int(instruction.rt, 2)] = ~(
            registers[int(instruction.rs, 2)] | complement_to_int(immediate))
        PC[0] += 4


def simulation_SLT(instruction, PC, data_address, registers, memory):
    if instruction.category == 0:
        if registers[int(instruction.rs, 2)] < registers[int(instruction.rt, 2)]:
            registers[int(instruction.rd, 2)] = 1
        else:
            registers[int(instruction.rd, 2)] = 0
        PC[0] += 4
    else:
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        if registers[int(instruction.rs, 2)] < complement_to_int(immediate):
            registers[int(instruction.rt, 2)] = 1
        else:
            registers[int(instruction.rt, 2)] = 0
        PC[0] += 4