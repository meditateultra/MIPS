from utils import *
from simulation import *


class ScoreBoard:
    def __init__(self, instructions, PC, data_address, registers, memory, start_address, simulation_file, count):
        self.instructions = instructions
        self.PC = PC
        self.data_address = data_address
        self.registers = registers
        self.memory = memory
        self.start_address = start_address
        self.simulation_file = simulation_file
        self.count = count

        self.pre_issue = []
        self.pre_mem = []
        self.pre_alu = []
        self.pre_alub = []
        self.post_mem = []
        self.post_alu = []
        self.post_alub = []

        # 寄存器是否准备就绪 0代表准备就绪
        self.register_status = [0] * 32
        self.store_ready = [True] * 32

        # 用于判断WAW，RAW， WAR
        # register_write存当前写该寄存器的指令在pre_issue中的index
        # register_read存当前读该寄存器的指令在pre_issue中的index
        self.register_write = [4] * 32
        self.register_read = [4] * 32
        # 所有跳转指令
        self.jump_instruction = ['J', 'JR', 'BEQ', 'BLTZ', 'BGTZ']

        self.is_break = False
        # IF是否被stall
        self.is_stall = False
        self.cycle = 1
        self.alub_cycle = 0

        # IF正在等待的指令
        self.if_wait_instruction = None
        # IF正在执行的跳转指令
        self.if_executed_instruction = None

    # 阻塞所有写指令涉及的目标寄存器
    def block_instruction(self, instruction):
        ins_name_str = get_instruction_name(instruction)
        if ins_name_str in ['ADDI', 'ANDI', 'ORI', 'XORI', 'LW']:
            rt = int(instruction.rt, 2)
            self.register_status[rt] += 1
        elif ins_name_str in ['SLL', 'SRL', 'SRA']:
            rd = int(instruction.rd, 2)
            self.register_status[rd] += 1
        elif ins_name_str in ['ADD', 'SUB', 'MUL', 'AND', 'NOR', 'SLT']:
            if instruction.category == 0:
                rd = int(instruction.rd, 2)
                self.register_status[rd] += 1
            else:
                rt = int(instruction.rt, 2)
                self.register_status[rt] += 1

    def instruction_step(self, instruction):
        '''
        执行跳转指令
        :param instruction:
        :return: True - 跳转成功  False - 跳转不成功
        '''
        ins_name_str = get_instruction_name(instruction)
        if ins_name_str == 'J' or ins_name_str == 'JR':
            eval(name_map_function[ins_name_str])(instruction, self.PC, self.data_address,
                                                  self.registers, self.memory)
            # 不需要什么条件，总会跳的
            self.is_stall = False
        elif ins_name_str == 'BLTZ' or ins_name_str == 'BGTZ':
            rs = int(instruction.rs, 2)
            # rs被阻塞着呢，因此要stall IF
            if self.register_status[rs] > 0:
                self.is_stall = True
                return False
            else:
                eval(name_map_function[ins_name_str])(instruction, self.PC, self.data_address,
                                                      self.registers, self.memory)
        elif ins_name_str == 'BEQ':
            rs = int(instruction.rs, 2)
            rt = int(instruction.rs, 2)
            # rs和rt只要有一个被阻塞，就要stall IF
            if self.register_status[rs] > 0 or not self.register_status[rs] > 0:
                self.is_stall = True
                return False
            else:
                eval(name_map_function[ins_name_str])(instruction, self.PC, self.data_address,
                                                      self.registers, self.memory)
        return True

    def instruction_fetch(self):
        if_instructions = []
        self.if_executed_instruction = None
        # IF被stall了
        if self.is_stall:
            instruction = self.if_wait_instruction
            # print(str(instruction))
            # 是否成功跳转
            is_step = self.instruction_step(instruction)
            # 成功跳转，说明IF当前正在执行跳转指令
            if is_step:
                self.is_stall = False
                self.if_executed_instruction = instruction
                self.if_wait_instruction = None
        # IF没有被stall，正常获取指令
        else:
            num = 0
            # 每次最多获取两条&&pre_issue要有空位
            while num < min(2, 4 - len(self.pre_issue)):
                index = int((self.PC[0] - self.start_address) / 4)
                instruction = self.instructions[index]
                ins_name_str = get_instruction_name(instruction)
                # BREAK中断所有操作
                if ins_name_str == 'BREAK':
                    self.if_executed_instruction = instruction
                    self.is_break = True
                    self.is_stall = True
                    break
                elif ins_name_str in self.jump_instruction:
                    is_step = self.instruction_step(instruction)
                    # 成功跳转，说明IF当前正在执行跳转指令
                    if is_step:
                        self.is_stall = False
                        self.if_executed_instruction = instruction
                        self.if_wait_instruction = None
                        break
                    # 没有跳转成功，stall IF
                    else:
                        self.is_stall = True
                        self.if_executed_instruction = None
                        self.if_wait_instruction = instruction
                        break
                else:
                    if_instructions.append(instruction)
                    # 阻塞相应寄存器
                    self.block_instruction(instruction)
                    self.PC[0] += 4
                    num += 1
        return if_instructions

    def is_instruction_ready(self, instruction, index):
        rs = int(instruction.rs, 2)
        rt = int(instruction.rt, 2)
        rd = int(instruction.rd, 2)
        ins_name_str = get_instruction_name(instruction)
        ready = False
        if ins_name_str in ['ADD', 'SUB', 'MUL', 'AND', 'NOR', 'SLT']:
            if instruction.category == 0:
                # RAW, RAW, WAR, WAW
                if self.register_write[rs] >= index and self.register_write[rt] >= index \
                        and self.register_read[rd] >= index and self.register_write[rd] >= index \
                        and self.store_ready[rd]:
                    ready = True
                # 更新index
                if self.register_write[rd] > index:
                    self.register_write[rd] = index
                if self.register_read[rs] > index:
                    self.register_read[rs] = index
                if self.register_read[rt] > index:
                    self.register_read[rt] = index
            else:
                # RAW, WAR, WAW
                if self.register_write[rs] >= index and self.register_read[rt] >= index \
                        and self.register_write[rt] >= index and self.store_ready[rt]:
                    ready = True
                if self.register_write[rt] > index:
                    self.register_write[rt] = index
                if self.register_read[rs] > index:
                    self.register_read[rs] = index
        elif ins_name_str in ['SLL', 'SRL', 'SRA']:
            # RAW, WAR, WAW
            if self.register_write[rt] >= index and self.register_read[rd] >= index \
                    and self.register_write[rd] >= index and self.store_ready[rd]:
                ready = True
            if self.register_read[rt] > index:
                self.register_read[rt] = index
            if self.register_write[rd] > index:
                self.register_write[rd] = index
        elif ins_name_str == 'LW':
            # RAW, WAR, WAW, rt不在执行单位中被写入
            if self.register_write[rs] >= index and self.register_read[rt] >= index \
                    and self.register_write[rt] >= index and self.store_ready[rt]:
                ready = True
            if self.register_write[rt] > index:
                self.register_write[rt] = index
            if self.register_read[rs] > index:
                self.register_read[rs] = index
        elif ins_name_str == 'SW':
            # RAW, RAW
            if self.register_write[rt] >= index and self.register_write[rs] >= index:
                ready = True
            if self.register_read[rt] > index:
                self.register_read[rt] = index
            if self.register_read[rs] > index:
                self.register_read[rs] = index
        return ready

    def instruction_issue(self):
        pre_alu_copy = []
        pre_alub_copy = []
        pre_mem_copy = []
        remove_instructions = []
        # 一次最多发两条
        num = 0
        for i, instruction in enumerate(self.pre_issue):
            if num == 2:
                break
            rs = int(instruction.rs, 2)
            rt = int(instruction.rt, 2)
            rd = int(instruction.rd, 2)
            ins_name_str = get_instruction_name(instruction)
            ins_arg = generate_instruction_str(ins_name_str, instruction)
            ready = self.is_instruction_ready(instruction, i)
            if ready:
                num += 1
                remove_instructions.append(instruction)
                # 进alub
                if ins_name_str in ['SLL', 'SRL', 'SRA', 'MUL'] and len(self.pre_alub) + len(pre_alub_copy) < 2:
                    if ins_name_str in ['SLL', 'SRL', 'SRA']:
                        if self.register_read[rt] >= i:
                            self.register_read[rt] = 4
                        self.register_write[rd] = -1
                        # rd要被写入，因此要锁住它
                        self.store_ready[rd] = False
                        # 要记录rt寄存器中的值
                        pre_alub_copy.append((instruction, self.registers[rt]))
                    else:
                        if instruction.category == 0:
                            if self.register_read[rt] >= i:
                                self.register_read[rt] = 4
                            if self.register_read[rs] >= i:
                                self.register_read[rs] = 4
                            self.register_write[rd] = -1
                            self.store_ready[rd] = False
                            # 要存储rs, rd中的值，否则会出现RAW
                            pre_alub_copy.append((instruction, self.registers[rs], self.registers[rt]))
                        else:
                            if self.register_read[rs] >= i:
                                self.register_read[rs] = 4
                            self.register_write[rt] = -1
                            self.store_ready[rt] = False
                            immediate = instruction.rd + instruction.shift_amount + instruction.func
                            pre_alub_copy.append((instruction, self.registers[rs], complement_to_int(immediate)))
                # 进mem
                elif ins_name_str in ['LW', 'SW'] and len(self.pre_mem) + len(pre_mem_copy) < 2:
                    if ins_name_str == 'LW':
                        if self.register_read[rs] > i:
                            self.register_read[rs] = 4
                        self.register_write[rt] = -1
                        self.store_ready[rt] = False
                        # 要记录一下base寄存器中的值, 不记录可能出现RAW
                        pre_mem_copy.append((instruction, self.registers[rs]))
                    else:
                        if self.register_read[rt] >= i:
                            self.register_read[rt] = 4
                        if self.register_read[rs] >= i:
                            self.register_read[rs] = 4
                        pre_mem_copy.append((instruction, self.registers[rs], self.registers[rt]))
                # 进alu
                elif ins_name_str in ['ADD', 'SUB', 'AND', 'NOR', 'SLT'] and len(self.pre_alu) + len(pre_alu_copy) < 2:
                    if instruction.category == 0:
                        if self.register_read[rt] >= i:
                            self.register_read[rt] = 4
                        if self.register_read[rs] >= i:
                            self.register_read[rs] = 4
                        self.register_write[rd] = -1
                        self.store_ready[rd] = False
                        # 要存储rs, rd中的值，否则会出现RAW
                        pre_alu_copy.append((instruction, self.registers[rs], self.registers[rt]))
                    else:
                        if self.register_read[rs] >= i:
                            self.register_read[rs] = 4
                        self.register_write[rt] = -1
                        self.store_ready[rt] = False
                        immediate = instruction.rd + instruction.shift_amount + instruction.func
                        pre_alu_copy.append((instruction, self.registers[rs], complement_to_int(immediate)))
                else:
                    # 指令塞不进去
                    remove_instructions.remove(instruction)
        # 需要删除的指令
        for instruction in remove_instructions:
            self.pre_issue.remove(instruction)
        return pre_alu_copy, pre_alub_copy, pre_mem_copy

    def alu_calculate(self):
        post_alu_copy = []
        if len(self.pre_alu) > 0:
            instruction, x, y = self.pre_alu.pop(0)
            ins_name_str = get_instruction_name(instruction)
            result = 0
            if ins_name_str == 'ADD':
                result = x + y
            elif ins_name_str == 'SUB':
                result = x - y
            elif ins_name_str == 'AND':
                result = x & y
            elif ins_name_str == 'NOR':
                result = ~(x | y)
            elif ins_name_str == 'SLT':
                result = 1 if x < y else 0
            post_alu_copy.append((instruction, result))
        return post_alu_copy

    def alub_calculate(self):
        post_alub_copy = []
        if len(self.pre_alub) > 0:
            self.alub_cycle += 1
            if self.alub_cycle == 2:
                self.alub_cycle = 0
                result = 0
                temp = self.pre_alub.pop(0)
                instruction = temp[0]
                # SRL, SLL, SRA
                if len(temp) == 2:
                    x = temp[1]
                    ins_name_str = get_instruction_name(instruction)
                    if ins_name_str == 'SLL':
                        result = (x << int(instruction.shift_amount, 2)) & (2 ** 32 - 1)
                    elif ins_name_str == 'SRL':
                        sa = complement_to_int(instruction.shift_amount)
                        s = 32 - sa
                        res = int_to_complement(x)[:s]
                        res = '0' * sa + res
                        result = complement_to_int(res)
                    elif ins_name_str == 'SRA':
                        sa = complement_to_int(instruction.shift_amount)
                        s = 32 - sa
                        res = int_to_complement(x)[:s]
                        res = res[0] * sa + res
                        result = complement_to_int(res)
                else:
                    x = temp[1]
                    y = temp[2]
                    result = x * y
                post_alub_copy.append((instruction, result))
        return post_alub_copy

    def mem_calculate(self):
        post_mem_copy = []
        if len(self.pre_mem) > 0:
            temp = self.pre_mem.pop(0)
            instruction = temp[0]
            x = temp[1]
            offset = instruction.rd + instruction.shift_amount + instruction.func
            valid_address = int((x + complement_to_int(offset) - self.data_address) / 4)
            result = 0
            if len(temp) == 2:
                result = self.memory[int(valid_address)]
                post_mem_copy.append((instruction, result))
            else:
                y = temp[2]
                self.memory[int(valid_address)] = y
        return post_mem_copy

    def wb_calculate(self):
        if len(self.post_alu) == 1:
            instruction, result = self.post_alu.pop(0)
            rs = int(instruction.rs, 2)
            rt = int(instruction.rt, 2)
            rd = int(instruction.rd, 2)
            if instruction.category == 0:
                self.registers[rd] = result
                self.register_status[rd] -= 1
                self.register_write[rd] = 4
                self.store_ready[rd] = True
            else:
                self.registers[rt] = result
                self.register_status[rt] -= 1
                self.register_write[rt] = 4
                self.store_ready[rt] = True
        if len(self.post_alub) == 1:
            instruction, result = self.post_alub.pop(0)
            rs = int(instruction.rs, 2)
            rt = int(instruction.rt, 2)
            rd = int(instruction.rd, 2)
            ins_name_str = get_instruction_name(instruction)
            if ins_name_str == 'MUL' and instruction.category == 1:
                self.registers[rt] = result
                self.register_status[rt] -= 1
                self.register_write[rt] = 4
                self.store_ready[rt] = True
            else:
                self.registers[rd] = result
                self.register_status[rd] -= 1
                self.register_write[rd] = 4
                self.store_ready[rd] = True
        if len(self.post_mem) == 1:
            instruction, result = self.post_mem.pop(0)
            rs = int(instruction.rs, 2)
            rt = int(instruction.rt, 2)
            rd = int(instruction.rd, 2)
            self.registers[rt] = result
            self.register_status[rt] -= 1
            self.register_write[rt] = 4
            self.store_ready[rt] = True

    # 输出函数
    def output(self):
        output_str = '--------------------\n'
        output_str += 'Cycle:' + str(self.cycle) + '\n'
        output_str += '\n'

        # IF Unit
        output_str += 'IF Unit:\n'

        # Waiting Instruction
        output_str += '\tWaiting Instruction: '
        if self.if_wait_instruction:
            instruction = self.if_wait_instruction
            ins_name_str = get_instruction_name(instruction)
            ins_args_str = generate_instruction_str(ins_name_str, instruction)

            output_str += ins_name_str + '\t' + ins_args_str + '\n'
        else:
            output_str += '\n'

        # Executed Instruction
        output_str += '\tExecuted Instruction: '
        if self.if_executed_instruction:
            instruction = self.if_executed_instruction
            ins_name_str = get_instruction_name(instruction)
            ins_args_str = generate_instruction_str(ins_name_str, instruction)

            if ins_name_str == 'BREAK':
                output_str += ins_name_str + '\n'
            else:
                output_str += ins_name_str + '\t' + ins_args_str + '\n'
        else:
            output_str += '\n'

        # Pre-Issue Buffer
        output_str += 'Pre-Issue Buffer:\n'
        for i in range(4):
            temp = ''
            if i < len(self.pre_issue):
                instruction = self.pre_issue[i]
                ins_name_str = get_instruction_name(instruction)
                ins_args_str = generate_instruction_str(ins_name_str, instruction)
                temp = '[' + ins_name_str + '\t' + ins_args_str + ']'
            output_str += '\tEntry ' + str(i) + ':' + temp + '\n'

        # Pre-ALU Queue
        output_str += 'Pre-ALU Queue:\n'
        for i in range(2):
            temp = ''
            if i < len(self.pre_alu):
                instruction = self.pre_alu[i][0]
                ins_name_str = get_instruction_name(instruction)
                ins_args_str = generate_instruction_str(ins_name_str, instruction)
                temp = '[' + ins_name_str + '\t' + ins_args_str + ']'
            output_str += '\tEntry ' + str(i) + ':' + temp + '\n'

        # Post-ALU Buffer
        output_str += 'Post-ALU Buffer:'
        if len(self.post_alu) > 0:
            instruction = self.post_alu[0][0]
            ins_name_str = get_instruction_name(instruction)
            ins_args_str = generate_instruction_str(ins_name_str, instruction)
            output_str += '[' + ins_name_str + '\t' + ins_args_str + ']'
        output_str += '\n'

        # Pre-ALUB Queue
        output_str += 'Pre-ALUB Queue:\n'
        for i in range(2):
            temp = ''
            if i < len(self.pre_alub):
                instruction = self.pre_alub[i][0]
                ins_name_str = get_instruction_name(instruction)
                ins_args_str = generate_instruction_str(ins_name_str, instruction)
                temp = '[' + ins_name_str + '\t' + ins_args_str + ']'
            output_str += '\tEntry ' + str(i) + ':' + temp + '\n'

        # Post-ALUB Buffer
        output_str += 'Post-ALUB Buffer:'
        if len(self.post_alub) > 0:
            instruction = self.post_alub[0][0]
            ins_name_str = get_instruction_name(instruction)
            ins_args_str = generate_instruction_str(ins_name_str, instruction)
            output_str += '[' + ins_name_str + '\t' + ins_args_str + ']'
        output_str += '\n'

        # Pre-MEM Queue
        output_str += 'Pre-MEM Queue:\n'
        for i in range(2):
            temp = ''
            if i < len(self.pre_mem):
                instruction = self.pre_mem[i][0]
                ins_name_str = get_instruction_name(instruction)
                ins_args_str = generate_instruction_str(ins_name_str, instruction)
                temp = '[' + ins_name_str + '\t' + ins_args_str + ']'
            output_str += '\tEntry ' + str(i) + ':' + temp + '\n'

        # Post-MEM Buffer
        output_str += 'Post-MEM Buffer:'
        if len(self.post_mem) > 0:
            instruction = self.post_mem[0][0]
            ins_name_str = get_instruction_name(instruction)
            ins_args_str = generate_instruction_str(ins_name_str, instruction)
            output_str += '[' + ins_name_str + '\t' + ins_args_str + ']'
        output_str += '\n'

        # registers
        output_str += '\nRegisters'

        for i in range(32):
            if i % 8 == 0:
                if i / 8 == 0:
                    output_str += '\nR00:'
                if i / 8 == 1:
                    output_str += '\nR08:'
                if i / 8 == 2:
                    output_str += '\nR16:'
                if i / 8 == 3:
                    output_str += '\nR24:'
            output_str += '\t' + str(self.registers[i])
        output_str += '\n\n'

        # data segement
        output_str += 'Data' + '\n'

        row = 0
        col = 0

        for i in range(self.count):
            mem = self.memory[i]
            if col == 0:
                output_str += str(self.data_address + row * 32) + ':\t'
            col += 1
            if col == 8:
                output_str += str(mem) + '\n'
                col = 0
                row += 1
            else:
                output_str += str(mem) + '\t'

        # 不足的补0
        while col > 0 and col < 8:
            col += 1
            if col == 8:
                output_str += '0\n'
            else:
                output_str += '0\t'

        return output_str

    def simulation(self):
        output = ''
        while True:
            if_instructions = self.instruction_fetch()
            pre_alu, pre_alub, pre_mem = self.instruction_issue()
            post_alu = self.alu_calculate()
            post_alub = self.alub_calculate()
            post_mem = self.mem_calculate()
            self.wb_calculate()

            self.pre_issue.extend(if_instructions)
            self.pre_alu.extend(pre_alu)
            self.pre_alub.extend(pre_alub)
            self.pre_mem.extend(pre_mem)
            self.post_alu.extend(post_alu)
            self.post_alub.extend(post_alub)
            self.post_mem.extend(post_mem)

            output += self.output()
            self.cycle += 1

            if self.is_break:
                break
        self.simulation_file.write(output)
