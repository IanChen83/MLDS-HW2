# import cPickle
# import random
# from sys import stdout
# from collections import deque
#
# from Testbench.command import status
# from util import *
#
# __author__ = 'patrickchen'
#
#
# class TestBench:
#     def __init__(self):
#         self.file = {
#             "train_input":  "train_sub.ark",
#             "train_answer": "answer_map_sub.txt",
#             "test_input":   "test_sub.ark",
#             "test_output":  "test_ans.csv"
#         }
#         self.train_input = []
#         self.train_answer = []
#         self.test_data = []
#         self.W_parm = None
#         self.B_parm = None
#
#         self.training = {
#             "modified": False,
#             "correct": 0,
#             "total": 0,
#             "current": 0,
#             "cost": [],
#             "acc": []
#         }
#         self.config = {
#             "layer":            [512],
#             "input_dimension":  69,
#             "output_dimension": 48,
#             "batch_num":        3,
#             "training_segment": 400,
#             "learning_rate":    0.01,
#             "momentum":         False,
#             "adagrad":          True,
#             "softmax":          False
#         }
#
#         self.model = None
#
#         self.ans_type = [
#             "aa", "ae", "ah", "ao", "aw", "ax", "ay", "b", "ch", "cl", "d",
#             "dh", "dx", "eh", "el", "en", "epi", "er", "ey", "f", "g", "hh",
#             "ih", "ix", "iy", "jh", "k", "l", "m", "ng", "n", "ow", "oy", "p",
#             "r", "sh", "sil", "s", "th", "t", "uh", "uw", "vcl", "v", "w",
#             "y", "zh", "z"
#         ]
#         self.prompt = ""
#         self.command_queue = deque()
#
#     def command_loop(self):
#         while True:
#             if len(self.command_queue) != 0:
#                 self.__exec_one_command__(self.command_queue.popleft())
#             self.prompt = "Enter a command:" if self.training["modified"] is False else "Enter a command[Modified]:"
#
#             command = raw_input(BColors.BLUE + self.prompt + BColors.END)
#             self.__exec_one_command__(command)
#
#     def create_model(self):
#         self.model = ModelFactory(
#             self.config["input_dimension"],
#             self.config["output_dimension"],
#             self.config["layer"],
#             self.config["batch_num"],
#             self.config["learning_rate"]
#         )
#
#     def __exec_one_command__(self, cmd):
#         dispatcher = {
#             "script":   self.script,
#             "status":   status.run,
#             "train":    self.train,
#             "load":     self.load,
#             "run":      self.run,
#             "output":   self.output,
#             "set":      self.set,
#             "quit":     TestBench.quit
#         }
#         commands = cmd.strip().split(';')
#
#         for x in commands:
#             m = x.strip().split()
#             if len(m) == 0:
#                 return
#             func = dispatcher.get(m[0], lambda (xx): TestBench.__do_nothing__)
#             func(m)
#
#     @staticmethod
#     def __do_nothing__(x):
#         print BColors.YELLOW, "Command not found.", BColors.END
#         pass
#
#     def script(self, param):
#         if len(param) == 1:
#             print BColors.RED + "ERROR: script command need 1 parameter"
#             return
#         try:
#             open_file = open(name=param[1], mode='r')
#             for command in open_file:
#                 self.command_queue.append(command)
#                 return
#         except IOError:
#             print BColors.RED + "ERROR: script file %s cannot be open" % param[1]
#
#     '''
#     #################### Training Functions ###################################
#     '''
#     @staticmethod
#     def update_display(cur, total, cur_epoch, cost, acc):
#         stdout.write(
#             "\r" + BColors.BLUE + "Progress: %d/%d (%.2f %%)" % (cur, total, float(cur) / total * 100) +
#             "\t" + "Current epoch: %d" % cur_epoch +
#             "\t" + "Cost: %.2f" % cost +
#             "\t" + "ACC: %.2f" % float(acc)
#         )
#         pass
#
#     def train(self, param):
#         epoch = 1
#         if len(param) > 1 and param[1].isdigit():
#             epoch = int(param[1])
#         self.train_data(epoch)
#
#     def train_data(self, epoch):
#         train_times = self.config["training_segment"] * epoch
#         if len(self.file["train_input"]) == 0:
#             self.load_train_input_data()
#         if self.model is None:
#             self.create_model()
#
#         train_batch_x = range(self.batch_number)
#         train_batch_y = range(self.batch_number)
#         i = 0
#         m = 0
#         cost = 0
#         acc = 0
#         while i < train_times:
#             for j in range(self.batch_number):
#                 num = random.randrange(0, self.config["train_segment"])
#                 train_batch_x[j], train_batch_y[j] = self.get_one_data(num), self.get_one_answer(num)
#
#             self.model.train_one(train_batch_x, train_batch_y)
#
#             i += self.batch_number
#             m += self.batch_number
#             if m > self.config["train_segment"]:
#                 cost = self.model.cost_function(train_batch_x, train_batch_y)
#                 acc = self._test(self.config["train_segment"])
#                 m -= self.config["train_segment"]
#
#             TestBench.update_display(cur=i,
#                                      total=train_times,
#                                      cur_epoch=i / self.config["train_segment"],
#                                      cost=cost,
#                                      acc=acc
#                                      )
#         stdout.write("\n")
#
#     '''
#     #################### Load Functions #######################################
#     '''
#
#     def load(self, param):
#         _id = 0
#         if len(param) > 1 and param[1].isdigit():
#             _id = int(param[1])
#         self.load_parameter(_id)
#
#     def load_parameter(self, _id):
#         filename_w = "parameter_W_%s.txt" % _id
#         filename_b = "parameter_B_%s.txt" % _id
#         filename_i = "parameter_I_%s.txt" % _id
#         my_print("Load parameters from parameter_X_%s.txt" % _id)
#
#         # TODO: Keep parameters of two test consistent
#
#         try:
#             i_parm_data = open(filename_i, 'r')
#             # self.update_parameter(i_parm_data)
#             if self.model is None:
#                 self.create_model()
#
#             w_parm_data = file(filename_w, 'rb')
#             b_parm_data = file(filename_b, 'rb')
#             w_parm = cPickle.load(w_parm_data)
#             b_parm = cPickle.load(b_parm_data)
#             self.model.load_parm(w_parm, b_parm)
#         except IOError:
#             my_print(BColors.RED + "File not found. Do nothing." + BColors.END)
#             return
#
#     def get_one_data(self, num):
#         return self.file["train_input"][num][1:self.config["input_dimension"] + 1]
#
#     def get_one_answer(self, num):
#         type_index = self.ans_type.index(str(self.train_answer[num][1].strip()))
#         g = [0] * self.config["output_dimension"]
#         g[type_index] = 1
#         return g
#
#     def run(self, param):
#         pass
#
#     '''
#     #################### Output Functions #####################################
#     '''
#
#     def output(self, param):
#         output_dispatcher = {
#             "progress": self._output_progress,
#             "csv": self._output_csv
#         }
#         if len(param) > 1:
#             output_command = output_dispatcher.get(param[1], lambda (xx): self.__do_nothing__)
#             output_command(param)
#             return
#         else:
#             print BColors.YELLOW + "Output parameter without any argument. Do nothing." + BColors.END
#             return
#         pass
#
#     def _output_progress(self, param):
#         pass
#
#     def _output_csv(self, param):
#         __id = 0
#         if len(param) > 2 and param[2].isdigit():
#             __id = int(param[2])
#
#         if len(self.test_data) == 0:
#             self.load_test_data()
#
#         filename_test = "test_ans_%d.txt" % __id
#         test_stream = open(filename_test, 'w')
#         my_print("Writing test answer data to %s" % filename_test)
#
#         test_stream.write('Id,Prediction\n')
#
#         for i in range(len(self.test_data)):
#             x, y = [self.get_one_data(i)], [[0] * self.config["output_dimension"]]
#             ya = self.model.y_evaluated_function(x, y)
#             value = "%s,%s" % (self.test_data[i][0], mapping_max(ya))
#             test_stream.write(value)
#             test_stream.write('\n')
#
#     def set(self, param):
#         set_dispatcher = {
#             "layer": self._set_layer,
#             "batch": self._set_batch
#         }
#         if len(param) > 2:
#             set_command = set_dispatcher.get(param[1], lambda: "Set parameter undefined. Do nothing")
#             set_command(param)
#             return
#         else:
#             print BColors.YELLOW + "Set parameter without any argument. Do nothing." + BColors.END
#             return
#
#     def _set_layer(self, param):
#         start_with = 2
#         layer = []
#         if self.model is not None:
#             if len(param) > 3 and param[2] == "force":
#                 start_with = 3
#             else:
#                 x = raw_input(BColors.YELLOW + "Set layer number without save parameters? [y/N]" + BColors.END)
#                 if x == "y" or x == 'Y':
#                     pass
#                     self.model = None
#                 else:
#                     return
#
#         for i in range(start_with, len(param)):
#             if param[i].isdigit():
#                 d = int(param[i])
#                 layer.append(d)
#
#         self.layer = layer
#         my_print("Layer are set to " + self.layer.__str__())
#
#     def _set_batch(self, param):
#         start_with = 2
#         if self.model is not None:
#             if len(param) > 3 and param[2] == "force":
#                 start_with = 3
#                 pass
#             else:
#                 x = raw_input(BColors.YELLOW + "Set layer number without save parameters? [y/N]" + BColors.END)
#                 if x == "y" or x == 'Y':
#                     self.model = None
#                     pass
#                 else:
#                     return
#
#         if param[start_with].isdigit():
#             d = int(param[start_with])
#             self.batch_number = d
#             my_print("Batch number is set to %d" % self.batch_number)
#
#     def save_parameter(self, _id):
#         f = file('parameter_W_%s.txt' % _id, 'wb')
#         cPickle.dump(self.model.W, f, protocol=cPickle.HIGHEST_PROTOCOL)
#         f.close()
#         f = file('parameter_B_%s.txt' % _id, 'wb')
#         cPickle.dump(self.model.B, f, protocol=cPickle.HIGHEST_PROTOCOL)
#         f.close()
#
#     @staticmethod
#     def quit(param):
#         if True is True:
#             if len(param) > 1 and param[1] == "force":
#                 exit()
#             else:
#                 x = raw_input(BColors.YELLOW + "Exit without save parameters? [y/N]" + BColors.END)
#                 if x == "y" or x == 'Y':
#                     exit()
#                 else:
#                     return
#         exit()
#
#     def load_train_input_data(self):
#         my_print("Load training input data from %s" % self.file["train_input"])
#         for line in open(self.file["train_input"], 'r'):
#             input_x = line.split()
#             input_x = [TestBench.float_convert(i) for i in input_x]
#             self.train_input.append(input_x)
#
#         my_print("Load training answer data from %s" % self.file["train_answer"])
#         for line in open(self.file["train_answer"], 'r'):
#             ans_x = line.split(',')
#             self.train_answer.append(ans_x)
#
#     def load_test_data(self):
#         for line in open(self.file["test_input"], 'r'):
#             test_x = line.split()
#             test_x = [TestBench.float_convert(x) for x in test_x]
#             self.test_data.append(test_x)
#
#     def save_test_data(self):
#         if len(self.file["test_input"]) == 0:
#             self.load_test_data()
#         my_print("Writing test answer data to %s", self.file["train_answer"])
#         test_stream = open(self.file["test_output"], 'w')
#         y = [0] * self.config["output_dimension"]
#         test_stream.write('Id,Prediction\n')
#         self.model.load_parm(self.W_parm, self.B_parm)
#
#         for i in range(len(self.test_data)):
#             ya = self.model.y_evaluated_function([self.test_data[i][1:self.config["input_dimension"]]], [y])
#             value = str(
#                 (self.test_data[i][0], mapping_max(ya))
#             )
#             test_stream.write(value)
#             test_stream.write('\n')
#         # print test.W_array[0].get_value()
#         return
#
#     def save_training_progress(self):
#         pass
#
#     def _test(self, training_segment):
#         err = 0
#         y = [0] * self.config["output_dimension"]
#         _1 = len(self.file["train_input"]) - training_segment
#         for m in range(_1):
#             # print self.file["train_input"]_data[training_segment + m][1:70]
#             xa = self.get_one_data(m + training_segment)
#             t = self.model.y_evaluated_function([xa], [self.get_one_answer(m + training_segment)])
#
#             if mapping_max(t) != str(self.train_answer[m + training_segment][1].strip()):
#
#                 err += 1
#             else:
#                 print mapping_max(t)
#                 print self.train_answer[m + training_segment][1].strip()
#
#                 # print [c.map(Ya)]
#                 # print [str(ans[m][1].split('\n')[0])]
#         return 1.0 - float(err / float(_1))
#
#     # def __run(self, batch):
#     #     training_segment = 1000000
#     #
#     #     batch_number = batch * 1000
#     #     X = None
#     #     Y = None
#     #     i = 0
#     #     acc = 0.0
#     #     W_new = []
#     #     B_new = []
#     #     c = MAP()
#     #     while True:
#     #         X = []
#     #         yy = []
#     #         for k in range(batch_number):
#     #             num = randrange(0, training_segment)
#     #             if i >= 1000000:  # i >= 1124823:
#     #                 i = 0
#     #                 err = 0.0
#     #                 for m in range(124823):
#     #                     Ya = self.model.y_evaluated_function([self.train[1000000 + m][1:70]], Y)[0]
#     #                     if [c.map(Ya)] != [str(self.ans[1000000 + m][1].split('\n')[0])]:
#     #                         err += 1
#     #                         # print [c.map(Ya)]
#     #                         # print [str(ans[m][1].split('\n')[0])]
#     #                 acc = 1.0 - err / 124823.0
#     #                 # print err
#     #                 print acc
#     #             type_idx = self.anstype.index(str(self.ans[num][1].split('\n')[0]))
#     #             y = [0] * 48
#     #             y[type_idx] = 1
#     #             yy.append(y)
#     #             X.append(self.train[num][1:70])
#     #             i += 1
#     #         Y = yy
#     #         if i % 10000 == 0:
#     #             print i
#     #             # print test.y_evaluated_function(X,Y)
#     #             # print [test.W_array[0].get_value(),test.W_array[1].get_value()]
#     #         self.model.train_one(X, Y)
#
#
#
#     @staticmethod
#     def get_correctness_ratio(correct, total):
#         return correct / total
#
#     @staticmethod
#     def float_convert(num):
#         try:
#             return float(num)
#         except ValueError:
#             return num
#
# '''
#     Test part
# '''
#
#
#
#
# '''
# train = []
# ans = []
# for line in train_data:
#     input_x = line.split()
#     input_x = [float_convert(i) for i in input_x]
#     train.append(input_x)
# for line in ans_data:
#     ans_x = line.split(',')
#     ans.append(ans_x)
# '''
#
# '''
#
# '''
# '''
#     Test part
# '''
#
# '''
# c = MAP()
# for i in range(400):
#     #print [train[i][1:70]]
#     Ya = test.y_evaluated_function([train[i][1:70]], Y)[0]
#     print c.map(Ya)
# '''
#
# # print test.train_one(X, Y)
#
# test = TestBench()
