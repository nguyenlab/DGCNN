import sys

import re


class GVertex:
    def __init__(self, id=None, name='', token='', toktype='ASM', content=None, incoming=None,outgoing =None):
        self.id = id
        self.name = name
        self.token = token
        self.toktype = toktype
        if content is None:
            content=[]
        if incoming is None:
            self.incoming =[]
        else:
            self.incoming = incoming
        if outgoing is None:
            self.outgoing =[]
        else:
            self.outgoing = outgoing
        self.content = content
    def getInDegree(self):
        return len(self.incoming)
    def getOutDegree(self):
        return len(self.outgoing)
    def getDegree(self):
        return len(self.incoming) + len(self.outgoing)
    def getData(self, toktypeDict, withOps = True):
        if not withOps:
            data =[self.token]
        else:
            data =[','.join(self.content)]
        if self.token in toktypeDict:
            data.append(toktypeDict[self.token])
        else:
            data.append('g_unknown')
        return data
    @staticmethod
    def getVertexFromVertexInfor(vinfor):
        return CFG.getVertexFromVertexInfor(vinfor)

    @staticmethod
    def getVertexFromEdge(vinfor):
        return CFG.getVertexFromEdge(vinfor)
    def show(self, buf = sys.stdout):
        buf.write(str(self.id) +'-'+ self.token+' (name='+self.name+', Content: '+ '_'.join(self.content)+')')
    def dump(self):
        return {'id':self.id,'name':self.name, 'token':self.token,'toktype':self.toktype, 'content':self.content}
class Graph:
    Vs =None
    Es =None
    def __init__(self, vs=None, es=None, label = -1):
        if vs is None:
            self.Vs ={}
        else:
            self.Vs = vs

        self.Es = []
        if es is not None:
            for (id1, id2) in es:
                self.addEdgebyID(id1, id2)
        self.label = label
    def addVetex(self, v=None):
        self.Vs[v.id] = v
    def addEdge(self, node1, node2):
        self.Es.append((node1.id, node2.id))
        if node2.id not in node1.outgoing:
            node1.outgoing.append(node2.id)
        if node1.id not in node2.incoming:
            node2.incoming.append(node1.id)

    def addEdgebyID(self, node1_id, node2_id):
        self.Es.append((node1_id, node2_id))
        node1 = self.Vs[node1_id]
        node2 = self.Vs[node2_id]
        if node2_id not in node1.outgoing:
            node1.outgoing.append(node2_id)
        if node1_id not in node2.incoming:
            node2.incoming.append(node1_id)

    def show(self, buf = sys.stdout):
        buf.write('Vertexes\n')
        for id in self.Vs:
            self.Vs[id].show(buf= buf)
            buf.write('\n')
        buf.write('Edges\n')
        for (v1, v2) in self.Es:
            buf.write(str(v1)+' --> '+ str(v2)+'\n')
    def toGraphViz(self):
        gviz = 'digraph finite_state_machine { node [shape = rectangle];\n'

        for id in self.Vs:
            gviz +='v{0}_{1}[label="{0}\\n{1}"];\n'.format(id,self.Vs[id].token)
        for (v1, v2) in self.Es:
            gviz += 'v{0}_{1} -> v{2}_{3};\n'.format(v1,self.Vs[v1].token,v2, self.Vs[v2].token)

        gviz+='}'
        return gviz
    def getVertexes(self):
        return self.Vs
    def getEdges(self):
        return self.Es
    def maxDegree(self):
        maxD =0
        for k, v in self.Vs.items():
            d= v.getDegree()
            if maxD < d:
                maxD = d
    def dump(self):
        vertexes=[]
        for v in self.Vs.values():
            vertexes.append(v.dump())
        edges =[]
        for (v1,v2) in self.Es:
            edges.append((v1, v2))
        return {'V':vertexes,'E':edges,'label':self.label}

    @staticmethod
    def load(dumped_obj):
        g = Graph()
        # get vertexes
        for v in dumped_obj['V']:
            g.addVetex(GVertex(id=v['id'], name=v['name'], token=v['token'], toktype=v['toktype'], content=v['content']))
        for v1_id, v2_id in dumped_obj['E']:
            g.addEdgebyID(v1_id, v2_id)
        g.label = dumped_obj['label']
        return g
    # @staticmethod
    # def loadGraph(file=''):
    #     nodename_dict ={}
    #     g = Graph()
    #
    #     return g
class CFG:
    val ='val'
    reg = 'reg'
    api = 'api'
    asm = 'asm'
    @staticmethod
    def getTypeName(ptype):
        bchar = ptype[0]
        if bchar =='%':
            return CFG.reg
        if bchar =='$':
            return CFG.val
        if ptype.__contains__('@'):
            return CFG.api
        if ptype.__contains__('%'):
            return CFG.reg
        val = ptype[2:]
        if re.match('(\d|[a-f])+', val, flags=0):
            return CFG.val
        return CFG.asm

    @staticmethod
    def getVertexFromEdge(vinfor):
        toktype = 'API'
        items = [vinfor]
        if vinfor.startswith('a0x'):
            toktype = 'ASM'
            idx1 = 11
            if re.match('a0x(\d|[a-f]){15,}', vinfor, flags=0):
                idx1 = 19

            instInfo = vinfor[idx1:]
            # print 'infor =', instInfo
            items = re.split('[_]*',instInfo)
        # print items
        vtok = items[0]
        # print items
        if vtok.startswith('0x'):
            vtok =CFG.val

        vtok = vtok.replace('@', '_').lower()
        content = [vtok]
        # print 'items=',items
        for idx in xrange(1, len(items)):
            if len(items[idx]) > 0:
                if items[idx].startswith('0x'):
                    content.append(CFG.val)
                else:
                    content.append(CFG.reg)
        # print vinfor, 'Content:', content
        vertex = GVertex(id=-1, name=vinfor, token=vtok, toktype=toktype, content=content)
        return vertex

    @staticmethod
    def getVertexFromVertexInfor(vinfor):
        vinfor = vinfor.lower()
        # print vinfor
        toktype ='API'
        if vinfor.startswith('a0x'):
            toktype ='ASM'
        # get vertex name
        idx = vinfor.index('[')
        vname = vinfor[:idx]

        vinfor = vinfor[idx+1:len(vinfor)-3] # remove 3 last tokens "];
        # get the information of instruction
        idx1 = vinfor.find('\\n')
        if idx1<0:
            idx1 = vinfor.find('="')
        idx2 = len(vinfor)
        if not vinfor.startswith('label="0x') < 0: # API or some special functions
            idx2 = vinfor.find('"',idx1+2)
            # raise Exception('error vertex infor:', vinfor)
        vinfor = vinfor[idx1+2: idx2]
        # print vinfor
        #detect group
        # if vinfor.__contains__('('):
        #     vinfor = re.split('[()]*', vinfor)
        #     for idx in range(0, len(vinfor)):
        #         vinfor[idx] = vinfor[idx].replace(',','')
        #     vinfor =' '.join(vinfor)
        #     # print 'new infor=' ,vinfor
        vinfor = vinfor.replace('(','')
        vinfor = vinfor.replace(',', '')
        items = re.split('[ ]*', vinfor)

        vtok = items[0]
        # print items
        content =[]

        for idx in xrange(1, len(items)):
            if len(items[idx])>0:
                itype = CFG.getTypeName(items[idx])
                if itype==CFG.asm or itype==CFG.api:
                    content.append(items[idx].replace('@','_'))
                else:
                    content.append(itype)

        vtok = vtok.replace('@','_')
        if vtok.startswith('0x'):
            vtok =CFG.val
        content.insert(0, vtok)
        vertex = GVertex(id=-1, name=vname, token=vtok, toktype=toktype, content= content)
        return vertex

# instruction ='a0x00424f11call_KeInitializeSpinLock[label="0x00424f11\\ncall KeInitializeSpinLock"];'
# vertex = CFG.getVertexFromVertexInfor(instruction)
# vertex = CFG.getVertexFromEdge('a0x01003f0bmovl_0xffffffffUINT32__4ebp_')
# print 'name:',vertex.name
# print 'token:',vertex.token
# print 'token type:',vertex.toktype
# print vertex.content
