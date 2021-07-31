import numpy as np
import pandas as pd
import itertools as it
from decimal import Decimal

# need to enforce rows summing to 1
# could be for either state transition or obs probs
class ProbabilityMatrix:
    def __init__(self, probabilities, row=None, col=None):
        # probabilities: m*n array where rows should sum to 1
        # row: (optional) a length m array of corresponding row names
        # col: (optional) a length n array of corresponding col names
        if row != None:
            assert len(probabilities) == len(row),\
                "The number of row labels must equal number of rows."
        if col != None:
            assert len(probabilities[0]) == len(col),\
                "The number of col labels must equal number of cols."
        for p in probabilities:
            assert len(p) == len(probabilities[0]),\
                "All rows should have the same length."
            assert abs(sum(p) - 1.0) < 1e-12,\
                "The sum of each row should be 1"

        if row == None:
            self.row = [i for i in range(len(probabilities))]
        else:
            self.row = row

        if col == None:
            self.col = [i for i in range(len(probabilities[0]))]
        else:
            self.col = col

        self.probabilities = np.array(probabilities)

    @property
    def df(self):
        # return a dataframe for the probability matrix
        return pd.DataFrame(self.probabilities, columns=self.col, index=self.row)
    
    def transition(self, start, end):
        # return a_start,end.
        # index() throws an error labels aren't found
        i = self.row.index(start)
        j = self.col.index(end)
        return self.probabilities[i][j]

    def diag(self, label):
        return pd.DataFrame(np.diag(self.df[label]), columns=self.row, index=self.row)
        # return ProbabilityMatrix(np.diag(self.df[label]), self.row, self.row)


    def __repr__(self):
        return self.df.to_string()

    def __matmul__(self, other):
        if isinstance(other, ProbabilityMatrix):
            return ProbabilityMatrix(self.probabilities @ other.probabilities,\
                self.row, other.col)
        # if isinstance(other, pd.DataFrame):
        #     return ProbabilityMatrix(self.probabilities @ other,\
        #         self.row, list(other.index))

################################################################################

A = ProbabilityMatrix([[0.7, 0.3],
                      [0.4, 0.6]],
                      row=["H", "C"],
                      col=["H", "C"])
B = ProbabilityMatrix([[0.1, 0.4, 0.5],
                      [0.7, 0.2, 0.1]],
                      row=["H", "C"],
                      col=["S", "M", "L"])

P = ProbabilityMatrix([[0.6, 0.4]],col=["H","C"])

# print(A)
# print(B)
# print(P)

O = [0,1,0,2]
O2 = ["S", "M", "S", "L"]
X = ["H","H","C","C"]

# Given L = (A,B,P), state sequence X, and observations O
# find the probability of observing O.
def P_XO(A,B,P,X,O):
    assert len(X) == len(O),\
        "Length of state seq and observations should be equal."

    prob = P.transition(0,X[0]) * B.transition(X[0],O[0])
    for i in range(1, len(X)):
        prob *= A.transition(X[i-1],X[i]) * B.transition(X[i],O[i])
    return prob

# print(P_XO(A,B,P,X,O2))

def state_seq_probabilities_tab(A,B,P,Q,O):
    # A,B,P: Markov model
    # Q: states
    # O: observations (length T)

    # returns a dataframe with the probability of observations O for each
    # possible state sequence

    perms = list(it.product(Q, repeat=len(O)))
    tab = []
    for perm in perms:
        tab.append([''.join(perm), float(round(Decimal(P_XO(A,B,P,perm,O)),6))])

    df = pd.DataFrame(tab, columns=["State","Probability"])
    total = df.sum()["Probability"]
    df["Normalized Probability"] = df.apply(lambda r: r["Probability"]/total, axis=1)

    return df

ss_df = state_seq_probabilities_tab(A,B,P, ["H","C"], O2)

# print("Table 1: State Sequence Probabilities")
# print(ss_df)
# print()

def hmm_probabilities_tab(A,B,P,Q,O):
    # A,B,P: Markov model
    # Q: state labels
    # O: observations (length T)

    # returns a dataframe with the probability of a given state at each time

    ss_df = state_seq_probabilities_tab(A,B,P,Q,O)
    df = pd.DataFrame([[]],index=Q)
    for i in range(len(O)):
        probs = []
        for q in Q:
            q_df = ss_df[ss_df["State"].str[i] == q]
            probs.append(q_df.sum()["Normalized Probability"])
        df[i] = probs
    return df

hmm_prob = hmm_probabilities_tab(A,B,P,["H","C"],O2)

# print("Table 2: HMM Probabilities")
# print(hmm_prob)
# print()

def P_O_given_L(A,B,P,O):
    # A,B,P: Markov model
    # O: observations (length T)
    # given HMM lambda (ABP), return probability of observing O

    alpha = P.probabilities@B.diag(O[0])
    for i in range(1,len(O)):
        alpha @= A.probabilities
        alpha @= B.diag(O[i]).reset_index(drop=True)

    return alpha

def P_O_given_L2(A,B,P,O):
    # A,B,P: Markov model
    # O: observations (length T)
    # given HMM lambda (ABP), return probability of observing O
    alpha = P@pd.DataFrame(np.diag(B[O[0]]))

    for i in range(1,len(O)):
        alpha @= A
        alpha @= pd.DataFrame(np.diag(B[O[0]])).reset_index(drop=True)

    return float(alpha.sum(axis=1))

def X_given_LO(A,B,P,O):
    # A,B,P: Markov model
    # O: observations (length T)    
    # given HMM lambda (ABP), return most likely state seq X


    alpha = P.probabilities@B.diag(O[0])
    alpha_tab = pd.DataFrame(alpha)

    beta = pd.DataFrame(np.ones(P.df.shape[1]),index=P.df.columns)
    beta_tab = beta.transpose()

    for i in range(1,len(O)):
        alpha @= A.probabilities
        alpha @= B.diag(O[i]).reset_index(drop=True)
        alpha_tab = alpha_tab.append(pd.DataFrame(alpha), ignore_index=True)

        beta = A.probabilities @ B.diag(O[len(O)-i]).reset_index(drop=True) @ beta
        beta.index= P.df.columns
        beta_tab = beta.transpose().append(beta_tab, ignore_index=True)

    gamma_tab = alpha_tab*beta_tab
    gamma_tab = gamma_tab.div(gamma_tab.sum(axis=1)[0])

    return ','.join(list(gamma_tab.idxmax(axis=1)))

POL = P_O_given_L(A,B,P,O2)
# print("Problem 1 - Find P(O|λ):")
# print(POL)
# print(POL.sum(axis=1))
# print()

XLO = X_given_LO(A,B,P,O2)
# print("Problem 2 - Most likely state seq:")
# print(XLO)
# print()



def find_lambda(O,N,M):
    # O: sequence of observations
    # N: size of state space
    # M: size of observation space

    assert max(O) < M,\
        "Range of observations must be within M."

    # initialize and normalize A,B,P
    A = pd.DataFrame(np.random.rand(N,N))
    B = pd.DataFrame(np.random.rand(N,M))
    P = pd.DataFrame(np.random.rand(1,N))
    A += 10
    B += 10
    P += 10
    A = A.div(A.sum(axis=1), axis=0)
    B = B.div(B.sum(axis=1), axis=0)
    P = P.div(P.sum(axis=1), axis=0)
    # probability of getting O given this HMM
    p_O = P_O_given_L2(A,B,P,O)

    print("A:")
    print(A)
    print("B:")
    print(B)
    print("P:")
    print(P)
    print("P(O|λ):")
    print(p_O)

    for step in range(100):
        alpha = P@pd.DataFrame(np.diag(B[O[0]]))
        alpha_tab = pd.DataFrame(alpha)

        beta = pd.DataFrame(np.ones(P.shape[1]),index=P.columns)
        beta_tab = beta.transpose()

        for i in range(1,len(O)):
            alpha @= A
            alpha @= pd.DataFrame(np.diag(B[O[i]])).reset_index(drop=True)
            alpha_tab = alpha_tab.append(pd.DataFrame(alpha), ignore_index=True)

            beta = A @ pd.DataFrame(np.diag(B[O[len(O)-i]])).reset_index(drop=True) @ beta
            beta.index= P.columns
            beta_tab = beta.transpose().append(beta_tab, ignore_index=True)

        gamma_tab = alpha_tab*beta_tab
        gamma_tab = gamma_tab.div(gamma_tab.sum(axis=1)[0])

        #make di_gamma table (Tx(NxN array))
        di_gamma_tab = pd.DataFrame()
        for t in range(0,len(O)-1):
            # print("t = %d" % t)
            di_gamma = A.mul(np.array(alpha_tab[t:t+1]), axis=0)
            di_gamma @= pd.DataFrame(np.diag(B[O[t+1]])).reset_index(drop=True)
            di_gamma = di_gamma.mul(np.array(beta_tab[t+1:t+2]), axis=1)
            # print(di_gamma)
            # print(di_gamma.sum(axis=1))
            # di_gamma = di_gamma.div(float(p_O.sum(axis=1)))
            di_gamma = di_gamma.div(di_gamma.sum(axis=1), axis=0)
            # print(di_gamma)
            # print(gamma_tab[t:t+1])
            di_gamma = di_gamma.mul(np.array(gamma_tab[t:t+1]), axis=0)
            # print(di_gamma)
            # print(di_gamma.to_numpy().sum())
            cols = pd.MultiIndex.from_product([[t], [x for x in range(P.shape[1])]])
            # print(cols)
            di_gamma = pd.DataFrame(np.array(di_gamma), index=cols)
            di_gamma_tab = di_gamma_tab.append(di_gamma)
            # print(di_gamma_tab)
        
        print("====================== step %d ======================" % step)
        # print("alpha_tab:")
        # print(alpha_tab)
        # print("beta_tab:")
        # print(beta_tab)
        print("gamma_tab:")
        print(gamma_tab)
        # print("sum of gamma_tab cols to t-2:")
        sum_gamma_tab_t2 = gamma_tab[:-1].sum(axis=0)
        # print(sum_gamma_tab_t2)

        # print("sum of gamma_tab cols to t-1:")
        sum_gamma_tab_t1 = gamma_tab.sum(axis=0)
        # print(sum_gamma_tab_t1)


        # print("di_gamma_tab:")
        # print(di_gamma_tab)
        sum_di_gamma_tab = di_gamma_tab.sum(level=1)
        # print("sum of di_gamma_tab over t:")
        # print(sum_di_gamma_tab)
        # print("\n\n")

        P_comp = gamma_tab[:1]
        print("P_comp:")
        print(P_comp)

        A_comp = sum_di_gamma_tab.div(np.array(sum_gamma_tab_t2),axis=0)
        print("A_comp:")
        print(A_comp)


        # print(gamma_tab.iloc[[0,2]])
        print("B_comp:")
        B_comp = pd.DataFrame()
        for j in range(M): # cols
            # print("j: %d" %j)
            rows = [t for t, o_t in enumerate(O) if o_t == j]
            B_j = gamma_tab.iloc[rows].sum(axis=0).div(np.array(sum_gamma_tab_t1),axis=0)
            # print(B_j)
            B_comp[j] = B_j
            # print(B_comp)

            # B_j = pd.DataFrame(np.array(B_j), index=[j])
            # print(B_j)
            # B_comp = B_comp.append(B_j)
            # print(rows)
            # print(gamma_tab.iloc[rows])
            # print("sum:")
            # print(gamma_tab.iloc[rows].sum(axis=0))
            # print(sum_gamma_tab_t1)
            # print("div:")
            # print(gamma_tab.iloc[rows].sum(axis=0).div(np.array(sum_gamma_tab_t1),axis=0))
            # print()
        print(B_comp)

        p_O_comp = P_O_given_L2(A_comp,B_comp,P_comp,O)
        # print("p_O: %f" % float(p_O.sum(axis=1)))
        # print("p_O_comp: %f" % float(p_O_comp.sum(axis=1)))
        print("p_O: %f" % p_O)
        print("p_O_comp: %f" % p_O_comp)

        if (p_O >= p_O_comp):
            print("Re-estimated HMM was not better... T_T")
            break

        A = A_comp
        B = B_comp
        P = P_comp
        p_O = p_O_comp
        # print("test")
        # print(gamma_tab[2:3])
        # print(np.array([np.array(gamma_tab)]))


    # print("SUMMARY:")
    # print("A:")
    # print(A)
    # print("B:")
    # print(B)
    # print("P:")
    # print(P)
    # print("P(O|λ):")
    # print(p_O.sum(axis=1))

    return

print("Problem 3 - Find A,B,π:")
print(find_lambda([0,1,2,0,1,2,0,1,2,0,1,2,0,1,2],3,3))
