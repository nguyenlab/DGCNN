import re

from enum import Enum
import sys
sys.path.append('../GCNN')

from Graph import GVertex, Graph

# decltoks=['_def','_globl','_seh_proc','_seh_endproc']
decltoks=['_ident','_section','_size','_globl','_type','_text']
class InstType(Enum):
    asm =0
    label = 1
    decl = 2

class Instruction:
    def __init__(self, id =0, name = '', params= None, type=InstType.asm, previous =None, next = None): # type = instruction or label
        self.id = id
        self.name = name
        if params == None:
            self.params =[]
        else:
            self.params = params
        self.type = type
        if previous is not None:
            self.previous = previous
        else:
            self.previous =[]
        if next is not None:
            self.next = next;
        else:
            self.next =[]
    def show(self, buf = sys.stdout):
        # buf.write('('+str(self.type)+') '+ str(self.id)+'-'+ self.name+'-'+','.join(self.params))
        buf.write('(' + str(self.type.name) + ') ' + str(self.id) + '-' + self.name)
    @staticmethod
    def fromString(instString):
        inst = Instruction()
        if not instString.startswith('\t'):
            inst.type = InstType.label
        instString = instString.strip()
        items = re.split('[:; ,\t]*', instString)
        items = filter(lambda x: x != '', items)
        if len(items)==0:
            return None
        # print items
        inst.name = items[0]
        inst.params = items[1:]
        if inst.name in decltoks:
            inst.type = InstType.decl
        return inst
class Block:
    def __init__(self, name='', instructions= None):
        self.name = name
        if instructions is None:
            self.instructions = []
        else:
            self.instructions = instructions
    def addInstruction(self, inst):
        self.instructions.append(inst)
    def show(self, buf = sys.stdout):
        buf.write(self.name+':\n')
        for inst in self.instructions:
            buf.write('\t{0} {1}\n'.format(inst.name, ' '.join(inst.params)))


val = 'val'
reg = 'reg'
name = 'name'
def getParamType(ptype):
    if ptype.startswith('%') or ptype.__contains__('(%'):
        return reg
    if re.match('[-\$]*\d+$', ptype, flags=0) or ptype.startswith('"'):
        return val
    return name
def getBlocks(asmfile,ignoreDecl = True):
    f = open(asmfile, 'r')
    isheader = True
    idx = 0

    blocks = []
    b = None
    # get instructions
    for line in f.readlines():
        if line.startswith('\t') and isheader:  # remove the header
            continue
        isheader = False

        line = line.rstrip()
        line = line.replace('.', '_')

        inst = Instruction.fromString(line)
        if inst is None:
            continue
        inst.id = idx
        if ignoreDecl and inst.type == InstType.decl:
            continue
        if inst.type == InstType.label:  # A label == begin a block
            if b is not None:
                blocks.append(b)
            b = Block(name=inst.name)
        else:
            b.addInstruction(inst)
            idx += 1
    if b is not None:
        blocks.append(b)

    # refer label --> block
    blocks_dict = {}
    for bidx, b in reversed(list(enumerate(blocks))):
        # print b.name
        if len(b.instructions) > 0 or bidx == len(blocks) - 1:
            blocks_dict[b.name] = b
        else:

            blocks_dict[b.name] = blocks_dict[blocks[bidx + 1].name]
    return blocks, blocks_dict

def CFGfromASM(asmfile, ignoreDisjointMain = True ,ignoreDecl = True):
    blocks, blocks_dict = getBlocks(asmfile ,ignoreDecl)

    # construct edges
    for idx, b1 in enumerate(blocks):
        for inst_idx, inst in enumerate(b1.instructions):
            # add flow edge
            if inst_idx<len(b1.instructions)-1:
                next_inst = b1.instructions[inst_idx + 1]
                inst.next.append(next_inst.id)
                next_inst.previous.append(inst.id)

            params = inst.params
            if len(params)<=0:
                continue
            bname =  params[0]
            if params[0].startswith('$'):
                bname = params[0][1:]
            if bname in blocks_dict:
                b2 = blocks_dict[bname]
                if len(b2.instructions)==0:
                    continue
                # edge: current inst ---> first inst of the jumped block
                if inst.name.startswith('j') or inst.name=='call':
                    b2_first = b2.instructions[0]
                    b2_first.previous.append(inst.id)
                    inst.next.append(b2_first.id)
                else:
                    b2_first = b2.instructions[0]
                    b2_first.next.append(inst.id)
                    inst.previous.append(b2_first.id)

                if inst.name =="call" and inst_idx<len(b1.instructions)-1:
                    # last inst of the jumped block --> the next inst of the current inst
                    next_inst = b1.instructions[inst_idx+1]
                    b2_last = b2.instructions[-1]
                    b2_last.next.append(next_inst.id)
                    next_inst.previous.append(b2_last.id)

    # labels = {}
    # for idx, vid in enumerate(Vs.keys()):
    #     labels[vid] = idx
    # for vid1, vid2 in Es:
    #     oldlabel = labels[vid1]
    #     for vid, l in labels.items():
    #         if l == oldlabel:
    #             labels[vid] = labels[vid2]
    #         labels[vid1] = labels[vid2]
    #     print labels
    # mainlabel = labels[1]
    # connectedmain = []
    # for vid, l in labels.items():
    #     if l == mainlabel:
    #         connectedmain.append(vid)
    #generate graphs
    vertices = {}
    edges =[]
    for b in blocks:
        for inst in b.instructions:
            name = inst.name
            content = [name]
            for p in inst.params:
                content.append(getParamType(p))
            v = GVertex(id=inst.id, name=str(inst.id), token=name, toktype='ASM', content=content)
            vertices[v.id] = v
            for v2id in inst.next:
                edges.append((v.id, v2id))
    g = Graph(vertices, edges)
    return g


# # asmfile='D:/C_SourceCode/Code/testfunc.s'
# asmfile = 'Z:/Experiment/ASMCFG/test_code/code_S/loop_dowhile.c.s'
# g = CFGfromASM(asmfile)
# g = fromASM(asmfile)
# print g.Vs[1].incoming
# print g.Vs[1].outgoing
# g.show()
# print g.toGraphViz()
# instString = '  .def	__main;	.scl	2;	.type	32;	.endef'
# items = re.split('[:; ,\t]*', instString)
# items = filter(lambda x : x != '', items)
# print items[1:]