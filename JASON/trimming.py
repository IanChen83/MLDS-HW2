f = open('try2.csv','r')
f_map = open('48_idx_chr.map_b','r')

f_data = []
name = []
for line in f:
	temp = line.split(',')
	f_data.append(temp[1].split('\n')[0])
	a = temp[0].split('_')
	name.append(a)

f_map_data_elem_2 = []
f_map_data_elem_0 = []
for line in f_map:
	temp = line.split('\t')
	f_map_data_elem_2.append(temp[2].split('\n')[0])
	f_map_data_elem_0.append(temp[0].split('\n')[0])


f_trim = open('trim_ans_1111.csv','w')


f_trim.write('id,phone_sequence')
f_trim.write('\n')
name_flag = 0


for i in range( len(f_data) ):
	if i != 0:
		if name_flag == 0 :
			f_trim.write(name[i][0])
			f_trim.write('_')
			f_trim.write(name[i][1])
			f_trim.write(',')
			name_flag = 1
			sil_start = 0
		if i != len(f_data) -1:
			if(name[i][0]==name[i+1][0] and name[i][1]==name[i+1][1]):
				if f_data[i]!=f_data[i+1]:
					typeidx = f_map_data_elem_0.index(f_data[i])
					if sil_start==1:
						f_trim.write(f_map_data_elem_2[typeidx])
					else:
						sil_start=1

			else:
				typeidx = f_map_data_elem_0.index(f_data[i])
				if (f_data[i]!='sil'):
					f_trim.write(f_map_data_elem_2[typeidx])
				f_trim.write('\n')
				name_flag = 0
		else:
			typeidx = f_map_data_elem_0.index(f_data[i])
			f_trim.write(f_map_data_elem_2[typeidx])
				
f.close()
f_map.close()
f_trim.close()

			

