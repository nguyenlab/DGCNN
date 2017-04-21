import struct

# see http://blog.csdn.net/lesky/article/details/5727473 
# for reference
def write_binary(fname, W, B):
    f = file( fname, 'wb' )
    numW = struct.pack('i', len(W) )
    numB = struct.pack('i', len(B) )
    
    f.write( numW )
    f.write( numB )
    for i in xrange( len(W) ):
        tmp = struct.pack('f', W[i,0] )
        f.write( tmp )

    for i in xrange( len(B) ):
        tmp = struct.pack('f', B[i,0] )
        f.write(tmp)
        
    f.close()        