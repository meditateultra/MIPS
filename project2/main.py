from instruction import *
from simulation import *
from scoreboard import ScoreBoard


def generate_disassembly(instruction, PC, disassembly_file):
    ins_name_str = get_instruction_name(instruction)
    ins_args_str = generate_instruction_str(ins_name_str, instruction)

    if ins_name_str == 'BREAK':
        disassembly_file.write(instruction_split_str(instruction) + '\t' +
                               str(PC[0]) + '\t' + ins_name_str + '\n')
        return True
    disassembly_file.write(instruction_split_str(instruction) +
                           '\t' +
                           str(PC[0]) +
                           '\t' +
                           ins_name_str +
                           ' ' +
                           ins_args_str +
                           '\n')
    return False


def generate_simulation(
        instruction,
        PC,
        data_address,
        registers,
        memory,
        simulation_file,
        count,
        cnt):
    ins_name_str = get_instruction_name(instruction)
    ins_args_str = generate_instruction_str(ins_name_str, instruction)
    # ins_args_str = eval(name_map_function[ins_name_str][0])(instruction)

    simulation_file.write("--------------------\n")

    flag = False

    if ins_name_str != 'BREAK':
        simulation_file.write("Cycle:" +
                              str(cnt) +
                              '\t' +
                              str(PC[0]) +
                              '\t' +
                              ins_name_str +
                              '\t' +
                              ins_args_str +
                              '\n')
    else:
        simulation_file.write("Cycle:" + str(cnt) + '\t' +
                              str(PC[0]) + '\t' + ins_name_str + '\n')
        flag = True

    simulation_file.write('\n')

    eval(
        name_map_function[ins_name_str])(
        instruction,
        PC,
        data_address,
        registers,
        memory)

    simulation_file.write('Registers\n')
    sim_str = 'R00:'
    for i in range(16):
        sim_str += '\t' + str(registers[i])
    sim_str += '\n'
    simulation_file.write(sim_str)
    sim_str = 'R16:'
    for i in range(16, 32):
        sim_str += '\t' + str(registers[i])
    sim_str += '\n'
    simulation_file.write(sim_str)
    simulation_file.write('\n')
    simulation_file.write('Data\n')

    for i in range(0, count, 8):
        simulation_file.write(str(data_address) +
                              ':\t' +
                              str(memory[i]) +
                              '\t' +
                              str(memory[i +
                                         1]) +
                              '\t' +
                              str(memory[i +
                                         2]) +
                              '\t' +
                              str(memory[i +
                                         3]) +
                              '\t' +
                              str(memory[i +
                                         4]) +
                              '\t' +
                              str(memory[i +
                                         5]) +
                              '\t' +
                              str(memory[i +
                                         6]) +
                              '\t' +
                              str(memory[i +
                                         7]) +
                              '\n')
        data_address += 8 * 4

    simulation_file.write('\n')
    return flag


input_path = './sample.txt'

if __name__ == '__main__':
    registers = [0] * 32
    memory = [0] * 64
    ins_str = read_input_file(input_path)
    instructions = []
    for ins in ins_str:
        instructions.append(instruction(ins))
    start_address = 64
    PC = [start_address]
    disassembly_file = open('disassembly.txt', 'w')
    data_index = 0
    for ins in instructions:
        print(ins)
        flag = generate_disassembly(ins, PC, disassembly_file)
        PC[0] += 4
        data_index += 1
        if flag:
            break
    data_ins = instructions[data_index:]
    count = 0
    for data in data_ins:
        current_data_index = int((PC[0] - start_address - 4 * data_index) / 4)
        memory[current_data_index] = complement_to_int(str(data))
        disassembly_file.write(str(data) +
                               '\t' +
                               str(PC[0]) +
                               '\t' +
                               str(complement_to_int(str(data))) +
                               '\n')
        PC[0] += 4
        count += 1
    disassembly_file.close()
    simulation_file = open('simulation.txt', 'w')
    cnt = 1
    PC[0] = start_address
    data_address = start_address + 4 * data_index
    flag = False
    scoreboard = ScoreBoard(instructions, PC, data_address, registers, memory,
                            start_address, simulation_file, count)
    scoreboard.simulation()

    simulation_file.close()
