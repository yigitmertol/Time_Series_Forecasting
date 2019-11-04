import numpy as np
import pandas as pd
import random
import pyESN




class esn():

    def __init__(self,
                 input_dim,
                 output_dim=1):

        self.input_dim = input_dim


        self.ESN = pyESN.ESN(n_inputs=input_dim,
                        n_outputs=output_dim,
                        n_reservoir=50,
                        sparsity=0.9,
                        spectral_radius=1,
                        noise=5,
                        input_scaling=True,
                        teacher_forcing=True,
                        random_state=42)

    def fit_and_get_score(self, input_signals, target_signal_tr, target_signal_val):
        train_preds = self.ESN.fit(input_signals['tr'].values, target_signal_tr.values)
        train_err = np.sqrt(np.mean((train_preds.flatten() - target_signal_tr.values) ** 2))

        val_preds = self.ESN.predict(inputs=input_signals['val'].values)
        val_err = np.sqrt(np.mean((val_preds.flatten() - target_signal_val.values) ** 2))

        val_r2 = 10 - val_err / target_signal_val.std()
        tr_r2 = 10 - train_err / target_signal_tr.std()

        return val_r2, tr_r2



class evolution():

    def __init__(self,
                 df,
                 masks,
                 indiviudal_size,
                 col_target_signal='lr_vwa_pr',

                 tune_hyp_params=False,
                 simple=True,
                 col_number_penalty=None,

                 cross_over_prob=0.5,
                 mutation_prob=0.05,
                 population_size=10,
                 init_positive=0.2,
                 ratio_elites=0.05,

                 verbose=False
                 ):

        self.df = df
        self.masks = masks

        self.population_size = population_size
        self.individual_size = indiviudal_size

        self.hyper_params = tune_hyp_params
        self.simple = simple
        self.penalty = col_number_penalty


        self.cross_over_prob = cross_over_prob
        self.mutation_prob = mutation_prob
        self.ratio_initial_positives = init_positive
        self.elites_size = int(ratio_elites*self.population_size) if int(ratio_elites*self.population_size)%2==0 else int(ratio_elites*self.population_size)+1

        self.population = [None] * self.population_size
        self.roulette = [np.nan] * self.population_size

        self.target_signals = {}
        for data_div in ['tr', 'val', 'te']:
            self.target_signals[data_div] = self.df[self.masks[data_div]][col_target_signal][1:]

        self.verbose = verbose

    def init_population(self):
        self.elites = []
        self.population = [np.random.choice([True, False],
                                            size=self.individual_size,
                                            p=[self.ratio_initial_positives, 1-self.ratio_initial_positives]) for i in range(self.population_size)]

    def compute_fitness(self, individual, esn):

        # Prepare input signals (with the columns of this individudal) for training and validation
        input_signals = {}
        for data_div in ['tr', 'val', 'te']:
            input_signals[data_div] = self.df[self.masks[data_div]][self.df.columns.values[individual]][:-1]
            # add bias column
            input_signals['bias'] = 1

        # Get R^2 score
        val_score, tr_score = esn.fit_and_get_score(input_signals,
                                                    self.target_signals['tr'],
                                                    self.target_signals['val'])

        fitness = val_score if val_score>0 else 0
        return fitness

    def compute_fitness_all(self):
        self.esns = [esn(len(self.df.columns.values[self.population[i]])) for i in range(self.population_size)]
        self.roulette =[self.compute_fitness(self.population[i], self.esns[i]) for i in range(self.population_size)]

    def roulette_wheel_selection(self):
        self.elites=[]
        if self.elites_size > 0:
            for i in range(self.elites_size):
                best_ind_idx = np.argmax(self.roulette)
                self.elites.append(self.population[best_ind_idx])
                if i == 0 and self.verbose:
                    print('Best individuals score: ',self.roulette[int(best_ind_idx)])
                del self.roulette[int(best_ind_idx)]
                del self.population[int(best_ind_idx)]

        if sum(self.roulette) == 0:
            probs = None
        else:
            probs = self.roulette/sum(self.roulette)

        return np.random.choice(self.population_size-len(self.elites),
                                self.population_size-len(self.elites),
                                p=probs)

    def cross_over(self, ind1, ind2):
        cross_over_index = np.random.randint(0, self.individual_size)
        offspring1, offspring2 = ind1.copy(), ind2.copy()

        offspring1[:cross_over_index] = ind2[:cross_over_index]
        offspring2[:cross_over_index:] = ind1[:cross_over_index]

        return offspring1, offspring2

    def mutate(self, individual):
        # Just flip a random index from True to False or vice versa
        idx_flip = np.random.randint(0, self.individual_size)
        individual[idx_flip] = not individual[idx_flip]
        return individual

    def evolve(self, generations=1000):

        self.init_population()
        for gen in range(1, generations):
            if gen%1==0: print('\n Genereation: ',gen)

            # Compute fitness of individual
            self.compute_fitness_all()
            self.parents = self.roulette_wheel_selection()

            # Elites gete selected automatically o the next generation
            self.offspring = self.elites
            for i in range(0,len(self.population), 2):

                # Select fit parents for reproduction
                off1_ind, off2_ind = self.parents[i:i + 2]

                if np.random.random() < self.cross_over_prob:
                    off1, off2 = self.cross_over(self.population[off1_ind],
                                                 self.population[off2_ind])
                else:
                    off1, off2 = self.population[off1_ind], self.population[off2_ind]


                if np.random.random() < self.mutation_prob:
                    off1 = self.mutate(off1)
                if np.random.random() < self.mutation_prob:
                    off2 = self.mutate(off2)


                self.offspring.extend([off1, off2])


            self.population = self.offspring





