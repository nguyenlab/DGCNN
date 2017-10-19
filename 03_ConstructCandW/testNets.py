import gl
import pycparser

gl.reConstruct= False # reconstruct For, While, DoWhile
gl.ignoreDecl = False # Ignore declaration branches

import writeXY_Train_CV_Test as NetConstruct
text ='''

int visited[5][8];
int i,j;

int current_area = 0, max_area = 0;
int Arr[5][8]={};

void prepare_visited_map() {
 for(i=0;i<5;i++) {
 for(j=0;j<8;j++) visited[i][j] = 0;
 }
}

void calculate_largest_area(int x, int y) {
 if(visited[x][y]) return;

 if(x<0 || y<0 || x>=5 || y>=8) return;

 if(!Arr[x][y]) {
 visited[x][y] = 1;
 return;
 }

 current_area++;
 visited[x][y] = 1;

 calculate_largest_area(x,y-1);
 calculate_largest_area(x+1,y);
 calculate_largest_area(x,y+1);
 calculate_largest_area(x-1,y);
 }


int main() {
  for(i=0;i<5;i++) {
 for(j=0;j<8;j++) {
 prepare_visited_map();
 calculate_largest_area(i,j);
 if(current_area > max_area) max_area = current_area;
 }
 }
 printf("Max area is %d",max_area);

}



'''

text ='''
    int a = 3;

'''

#
# text ='''
# int main()
# {
#     int sum =0;
#     for (int i=0; i<10; i++)
#         sum = sum + i;
# }
# '''

parser = pycparser.c_parser.CParser()
ast = parser.parse(text=text)  # Parse code to AST
if gl.reConstruct:  # reconstruct braches of For, While, DoWhile
    ast.reConstruct()
print 'AST:'
ast.show()
ast_str = []
ast.toNewickFormat(value=ast_str)
ast_str = ''.join(ast_str)

print '\n\nNetworks:'
layers = NetConstruct.InitNetbyText(text = ast_str)
print 'Totally:', len(layers), 'layer(s)'
for l in layers:
    if hasattr(l,'bidx') and l.bidx is not None:
        print l.name, '\tBidx', l.bidx[0], '\tlen biases=', len(l.bidx)
    else:
        print l.name
    # print "    Up:"
    # for c in l.connectUp:
    #     if hasattr(c,'Widx'):
    #         print "        ", c.xlayer.name, " -> ", c.ylayer.name, \
    #             '(xnum= ', c.xnum, ', ynum= ', c.ynum,'), weights = ', len(c.Widx)
    #     else:
    #         print "        ", c.xlayer.name, " -> ", c.ylayer.name, \
    #             '(xnum= ', c.xnum, ', ynum= ', c.ynum, ')'

    print "    Down:"
    for c in l.connectDown:
        if hasattr(c,'Widx'):
            print "        ", c.xlayer.name, " -> ", '|', \
                '(xnum= ', c.xnum, ', ynum= ', c.ynum,'), Wid = ', c.Widx[0]
        else:
            print "        ", c.xlayer.name, " -> ", '|', \
                '(xnum= ', c.xnum, ', ynum= ', c.ynum, ')'

