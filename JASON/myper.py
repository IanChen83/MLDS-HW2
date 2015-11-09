import numpy
import sys

one = sys.argv[1]  
two = sys.argv[2]
ans_data = open(one,'r')
est_data = open(two,'r')
ans = ans_data.readlines()
est = est_data.readlines()

if len(ans) != len(est):
    print "ERR::two file contain different amount of data"
else:
    accumulate_err = 0
    accumulate_total = 0
    for i in range(len(ans) - 1):
        r = []
        h = []
        ans_single = ans[i+1].split(',')[1].split('\n')[0]
        est_single = est[i+1].split(',')[1].split('\n')[0]
        for j in range(len(ans_single)):
            r.append(ans_single[j])
        for k in range(len(est_single)):
            h.append(est_single[k])

        d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
        d = d.reshape((len(r)+1, len(h)+1))
        for i in range(len(r)+1):
            for j in range(len(h)+1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion    = d[i][j-1] + 1
                    deletion     = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

         #edit distance
        accumulate_err += d[len(r)][len(h)]
        accumulate_total += len(r)
        print r
    print 'total error:',accumulate_err
    print 'total character:',accumulate_total
    print 'per:',float(accumulate_err)/float(accumulate_total)