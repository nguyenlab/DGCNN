# -----------------------------------------------------------------
# pycparser: func_write.py
#
# Tiny example of rewriting a AST node
#
# Copyright (C) 2014, Akira Hayakawa
# License: BSD
# -----------------------------------------------------------------
from __future__ import print_function
from pycparser import parse_file
import sys
from pycparser import c_ast

from pycparser import c_parser

# text = r"""
# int func1(int b[100], int N)
# {
#     int sum =0;
#     int x = 2;
#     for(int i=0; i<N; i++)
# 	    sum = sum + b[i]*x;
#     return total;
# }
# int func(int m,int q)
# {
# 	int b[100];
#
# 	int i,j,k=0,p=0,sum=1;
# 	for (j=q;j<=m;j++)
# 	{
# 		if (j*j>m)
# 		{
# 			p=k;
# 			break;
# 		}
# 		if (m%j==0)
# 		{
# 			b[k]=j;
# 		    k++;
# 		}
#
# 	}
# 	if (k>2||p!=0&&m!=2)
# 	{
# 		for (i=0;i<p;i++)
# 		{
# 			sum+=func(m/b[i],b[i]);
# 		}
# 		return (sum);
# 	}
# 	else
# 		return (1);
# }
# main()
# {
# 	int i,j,k=0,n,m,b[100]={0},a[100]={0};
# 	scanf("%d",&n);
# 	for (i=0;i<n;i++)
# 	{
# 		scanf("%d",&m);
# 	    b[i]=func(m,2);
# 	}
# 	for (j=0;j<i;j++)
# 	{
# 		printf("%d\n",b[j]);
# 	}
# }
# """
import func_defs
from pycparser._ast_gen import ASTCodeGenerator


def GetMajorFunction(root):
    v = func_defs.FuncDefVisitor()
    v.visit(root)
    nodes = v.nodes;

    major = nodes[0]
    for cnode in nodes:
        if(major.NodeNum() < cnode.NodeNum()):
            major = cnode
    return major


text = """
typedef long long int ll;
int GCD(int a, int b)
{
    if (a == 0 || b == 0)
        return 0;
    if (a == b)
        return a;
    if (a > b)
        return GCD(a-b, b);
    return GCD(a, b-a);
}
int main()
{
    int t,a,b;
    cin>>t;
    while(t--)
    {
        cin>>a>>b;
        cout<<GCD(a,b)<<" "<<a*b/GCD(a,b)<<endl;
    }
    return 0;
}
"""
parser = c_parser.CParser()
ast = parser.parse(text)
#ast.reConstruct()
ast.show()
print (ast.NodeNum())

ast_str = []
ast.toNewickFormat(value=ast_str)
ast_str = ''.join(ast_str)
print (ast_str)
# visitor = c_ast.NodeVisitor()
# visitor.visit(ast)
#ast = GetMajorFunction(ast)
#ast.reConstruct()
# ast.show()
#count = ast.NodeNum()
#print("\n\nNode count =", count )
#ast.savepathsroot2leaf(rootpath='', f=sys.stdout, branchsepa = '\n')

# f = open('ast_data', 'a')
# ast.exporttofile(offset=0, attrnames=False, nodenames=False, _my_node_name='PVA',f = f)
# f.write('\n')
# for i in range(0,len(ast.ext)):
#     print ("\n AST ", i+1)
#     func = ast.ext[i];
#     func.show(offset=0, attrnames=False, nodenames=False, showcoord=False, _my_node_name='PVA')

# ast.traverse(offset=0, buf = sys.stdout, attrnames=False, nodenames=False, showcoord=False, _my_node_name='PVA')
# ast.show(buf= sys.stdout, offset=0 ,attrnames=False, nodenames=False, showcoord=False, _my_node_name='PVA')
# assign = ast.ext[0].body.block_items[0]
# assign.lvalue.name = "y"
# assign.rvalue.value = 2

# print("After:")
# ast.show(offset=2)
