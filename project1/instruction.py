class instruction:

    def __init__(self, instruction_string):
        self.category = int(instruction_string[0:1])
        self.op_code = instruction_string[0:6]
        self.rs = instruction_string[6:11]
        self.rt = instruction_string[11:16]
        self.rd = instruction_string[16:21]
        self.shift_amount = instruction_string[21:26]
        self.func = instruction_string[26:]
        self.ins_str = instruction_string

    def __str__(self):
        return self.ins_str