import pickle, torch
from core.models import Actor
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
import numpy as np



class Parameters:
    def __init__(self):
        self.state_dim = 415; self.action_dim = 19
        self.remote_base = "http://grader.crowdai.org:1730"
        self.pop_size = 5

args = Parameters()
model = Actor(args)
model.load_state_dict(torch.load('models/m4'))





class Seed_RE:


    def __init__(self, args, model):

        self.args = args
        self.task_pipes = [Pipe() for _ in range(args.pop_size)]
        self.result_pipes = [Pipe() for _ in range(args.pop_size)]
        self.manager = Manager()
        self.net_repo = Manager().list()
        self.net_repo.append(model)


        self.evo_workers = [Process(target=rollout_worker, args=(i, self.task_pipes[i][1], self.result_pipes[i][1], self.net_repo)) for i in range(args.pop_size)]

        for worker in self.evo_workers: worker.start()

        #Trackers
        self.eval_flag = [True for _ in range(args.pop_size)]

        self.seed_scores = []; self.promising = []
        self.current_seed = 0



    def train(self):



        ################ ROLLOUTS ##############
        #Start Rollouts
        for id in range(self.args.pop_size):
            if self.eval_flag[id]:
                self.task_pipes[id][0].send(self.current_seed)
                self.current_seed += 1
                self.eval_flag[id] = False


        ########## SOFT -JOIN ROLLOUTS ############
        while True:
            for i in range(self.args.pop_size):
                if self.result_pipes[i][0].poll():
                    entry = self.result_pipes[i][0].recv()
                    tmp_seed = entry[1]; fit = entry[2]
                    self.seed_scores.append(np.array([tmp_seed, fit]))
                    self.eval_flag[i] = True

                    if fit < 9850 and fit > 9835:
                        print('Score', fit, 'for seed', tmp_seed)
                        self.promising.append(np.array([tmp_seed, fit]))
            break






if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class

    agent = Seed_RE(args, model)

    for gen in range(1, 100000000):
        agent.train()

        if gen % 20 == 0:
            handle = open('seedre', "wb")
            pickle.dump(agent.seed_scores, handle)

            np.savetxt('seeds.txt', np.array(agent.seed_scores), fmt='%.3f', delimiter=',')
            np.savetxt('promising_seeds.txt', np.array(agent.promising), fmt='%.3f', delimiter=',')















