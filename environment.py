import numpy as np
import random
import math
from tqdm import tqdm_notebook
import skfuzzy as fuzz


class Peer:

    def __init__(self, id, type='good', malicious_behavior_rate=0.5, collective=False):
        self.id = id
        self.malicious_behavior_rate = malicious_behavior_rate
        self.type = type
        self.cats = set()
        self.collective = collective

    def buy(self, seller):
        mark = success = seller.sell()
        fake = False
        if self.type == 'malicious':
            if self.collective and seller.type == 'malicious':
                mark = True
            else:
                mark = not success
            fake = True
        return {'mark': mark, 'success': success, 'fake': fake}

    def buy_fuzzy(self, seller):
        mark, success = seller.sell_fuzzy()
        fake = False
        if self.type == 'malicious':
            if self.collective and seller.type == 'malicious':
                mark = 'very good'
            else:
                mark = 'very bad' if mark in ['normal', 'good', 'very good'] else 'very good'
            fake = True
        return {'mark': mark, 'success': success, 'fake': fake}

    def sell(self):
        if self.type == 'malicious':
            return False
        return True

    def sell_fuzzy(self):
        if self.type == 'malicious':
#             return np.random.choice(['very bad', 'bad']), False
            return 'very bad', False
        else:
            return np.random.choice(['good', 'very good']), True

    def rate(self, real_rating):
        pass

    def add_cat(self, cat):
        self.cats.add(cat)

    def add_cats(self, cats):
        self.cats.update(cats)


class SimpleEnv:

    def __init__(self, num_peers, malicious_rate, min_cat_peer_rate, num_cats,
                 malicious_behavior_rate=0.05,
                 min_cat_peer_num=3,
                 collective=False):
        self.min_cat_peer_num = min_cat_peer_num
        self.min_cat_peer_rate = min_cat_peer_rate
        self.num_cats = num_cats
        self.cats = []
        self.malicious_behavior_rate = malicious_behavior_rate
        self.malicious_rate = malicious_rate
        self.num_peers = num_peers
        self.interactions = []
        self.peers = []
        self.convergence = []
        self.simple_marks = np.zeros(self.num_peers)
        self.collective = collective

        malicious_num = math.ceil(self.num_peers * self.malicious_rate)
        for i in range(malicious_num):
            self.peers.append(Peer(id=i, type='malicious', malicious_behavior_rate=malicious_behavior_rate, collective=collective))
        for i in range(self.num_peers - malicious_num):
            self.peers.append(Peer(id=i + malicious_num, type='good'))

        for cat in range(self.num_cats):
            cat_len = math.ceil(min_cat_peer_rate * self.num_peers)
            peer_subset = set(np.random.choice(self.peers, cat_len, replace=False))
            for p in peer_subset:
                p.add_cat(cat)
            self.cats.append(peer_subset)

        all_cats = set(range(self.num_cats))
        for peer in self.peers:
            if len(peer.cats) < self.min_cat_peer_num:
                cats = set(np.random.choice(list(all_cats - peer.cats), size=self.min_cat_peer_num - len(peer.cats),
                                            replace=False))
                peer.add_cats(cats)
                for cat in cats:
                    self.cats[cat].add(peer)

    def simulate(self, n_inter: int):
        for i in tqdm_notebook(range(n_inter)):
            buyer, seller = self.choose_peers()

            interaction = interact(buyer, seller)
            c = 1 if interaction['mark'] else 0
            self.simple_marks[seller.id] += c
            self.interactions.append(interaction)

    def choose_peers(self):
        buyer = np.random.choice(self.peers)
        cat = np.random.choice(list(buyer.cats))
        seller = np.random.choice(list(self.cats[cat] - {buyer}))
        return buyer, seller


class EigenTrustEnv(SimpleEnv):

    def __init__(self, num_peers, malicious_rate, pre_trusted_rate,
                 min_cat_peer_rate,
                 num_cats,
                 trust_upd,
                 malicious_behavior_rate=1.,
                 min_cat_peer_num=3,
                 a=0.5,
                 collective=False):
        super().__init__(num_peers=num_peers,
                         malicious_rate=malicious_rate,
                         malicious_behavior_rate=malicious_behavior_rate,
                         min_cat_peer_rate=min_cat_peer_rate,
                         min_cat_peer_num=min_cat_peer_num,
                         num_cats=num_cats,
                         collective=collective)
        self.trust_upd = trust_upd
        pre_trusted_num = math.ceil(self.num_peers * pre_trusted_rate)
        self.pre_trusted = set()
        for peer in self.peers[-pre_trusted_num:]:
            self.pre_trusted.add(peer)
        self.pre_trusted_dist = \
            np.array([1 / len(self.pre_trusted) if peer in self.pre_trusted else 0 for peer in self.peers])
        self.reputation = np.zeros(self.num_peers)
        self.local_marks = np.zeros((self.num_peers, self.num_peers))
        self.local_trust_matrix = np.zeros((self.num_peers, self.num_peers))
        self.a = a

    def choose_peers(self):
        buyer = np.random.choice(self.peers)
        cat = np.random.choice(list(buyer.cats))
        sellers, reps = [s for s in (self.cats[cat] - {buyer})], np.array([self.reputation[s.id] for s in
                                                                           (self.cats[cat] - {buyer})])
        with_zero = [s for s, r in zip(sellers, reps) if r == 0]

        # seller = sorted(sellers, key=lambda x: self.reputation[x.id], reverse=True)[0] #deterministic
        s = np.sum(reps)
        if len(with_zero) > 0:
            if np.random.rand() < 0.1 or s == 0:
                seller = np.random.choice(with_zero)
                return buyer, seller
        probs = [p / s for p in reps]
        seller = np.random.choice(sellers, p=probs)

        return buyer, seller

    def update_reputation(self):
        eps = 0.001
        t = self.pre_trusted_dist.copy()
        t_next = self.pre_trusted_dist.copy()
        dif = eps
        c = 0
        while dif >= eps:
            t_next = (1 - self.a) * self.local_trust_matrix.T.dot(t) + self.a * self.pre_trusted_dist
            d = t_next - t
            dif = np.linalg.norm(d)
            t = t_next
            c += 1
        self.convergence.append(c)
        self.reputation = t_next

    def update_local_trust_matrix(self, peer_i, peer_j, mark):
        self.local_marks[peer_i, peer_j] += mark
        s = np.sum([x for x in self.local_marks[peer_i] if x > 0])
        for j in range(self.num_peers):
            self.local_trust_matrix[peer_i, j] = max(self.local_marks[peer_i, j], 0) / s \
                if s != 0 \
                else self.pre_trusted_dist[peer_j]

    def simulate(self, n_inter: int):
        np.random.seed(42)
        random.seed(42)
        for i in tqdm_notebook(range(n_inter)):
            buyer, seller = self.choose_peers()

            interaction = interact(buyer, seller)
            mark = 1 if interaction['mark'] else -1
            self.update_local_trust_matrix(buyer.id, seller.id, mark)
            self.interactions.append(interaction)
            if (i + 1) % self.trust_upd == 0:
                self.update_reputation()


class HonestPeerEnv(EigenTrustEnv):

    def __init__(self, num_peers, malicious_rate, pre_trusted_rate,
                 min_cat_peer_rate,
                 num_cats,
                 trust_upd,
                 collective=False):
        super().__init__(num_peers=num_peers,
                         malicious_rate=malicious_rate,
                         pre_trusted_rate=pre_trusted_rate,
                         min_cat_peer_rate=min_cat_peer_rate,
                         num_cats=num_cats,
                         trust_upd=trust_upd,
                         collective=collective)

    def update_reputation(self):
        eps = 0.01
        t = self.pre_trusted_dist.copy()
        t_next = self.pre_trusted_dist.copy()
        dif = eps
        c = 0
        while dif >= eps:
            max_idx = np.argmax(self.reputation)
            max_rep = self.reputation[max_idx]
            a = max_rep if max_rep > 0.5 else 1 - max_rep
            t_next = (1 - a) * self.local_trust_matrix.T.dot(t) + a * self.pre_trusted_dist \
                if max_idx in self.pre_trusted \
                else a * self.local_trust_matrix.T.dot(t) + (1 - a) * self.pre_trusted_dist
            d = t_next - t
            dif = np.linalg.norm(d)
            t = t_next
            c += 1
        self.convergence.append(c)
        self.reputation = t_next


class PeerEigenTrustEnv(EigenTrustEnv):

    def __init__(self, num_peers, malicious_rate, min_cat_peer_rate, num_cats, trust_upd, pre_trusted_rate, a=0.5):
        super().__init__(num_peers=num_peers, malicious_rate=malicious_rate, min_cat_peer_rate=min_cat_peer_rate,
                         num_cats=num_cats, trust_upd=trust_upd, pre_trusted_rate=pre_trusted_rate, a=a)
        self.transactions = []
        self.connected_peers = []
        self.peer_trust_reputation = np.zeros(self.num_peers)
        for i in range(num_peers):
            self.transactions.append([])
            self.connected_peers.append(set())

    def add_transaction(self, peer_i, peer_j, mark):
        """

        :param peer_i: from
        :param peer_j: to
        :param mark:
        :return:
        """
        self.transactions[peer_j].append((peer_i, mark))
        self.connected_peers[peer_j].add(peer_i)

    def choose_peers(self):
        buyer = np.random.choice(self.peers)
        cat = np.random.choice(list(buyer.cats))
        sellers, reps = [s for s in (self.cats[cat] - {buyer})], np.array([self.peer_trust_reputation[s.id] for s in
                                                                           (self.cats[cat] - {buyer})])
        with_zero = [s for s, r in zip(sellers, reps) if r == 0]

        # seller = sorted(sellers, key=lambda x: self.reputation[x.id], reverse=True)[0] #deterministic
        s = np.sum(reps)
        if len(with_zero) > 0:
            if np.random.rand() < 0.1 or s == 0:
                seller = np.random.choice(with_zero)
                return buyer, seller
        probs = [p / s for p in reps]
        seller = np.random.choice(sellers, p=probs)

        return buyer, seller

    def update_reputation_peer(self):
        eps = 0.001
        t = np.ones(self.num_peers) * (1 / self.num_peers)
        t_next = np.ones(self.num_peers) * (1 / self.num_peers)
        dif = eps
        c = 0
        while dif >= eps:
            for v in range(self.num_peers):
                t_v = 0
                s = 0
                for i in self.connected_peers[v]:
                    s += t[i]
                if s != 0:
                    for tr in self.transactions[v]:
                        # tr[1] is a normalized amount of satisfaction from transaction,
                        # cf is a Credibility factor
                        cf = self.reputation[tr[0]]
                        t_v += tr[1] * cf
                t_next[v] = t_v
            d = t_next - t
            dif = np.linalg.norm(d)
            t = t_next
            c += 1
        self.convergence[-1] += c
        self.peer_trust_reputation = t_next

    def update_reputation_all(self):
        self.update_reputation()
        self.update_reputation_peer()

    def simulate(self, n_inter: int):
        np.random.seed(42)
        random.seed(42)
        for i in tqdm_notebook(range(n_inter)):
            buyer, seller = self.choose_peers()

            interaction = interact(buyer, seller)
            mark_p = 1 if interaction['mark'] else 0
            mark_e = 1 if interaction['mark'] else -1
            self.add_transaction(buyer.id, seller.id, mark_p)
            self.update_local_trust_matrix(buyer.id, seller.id, mark_e)
            self.interactions.append(interaction)
            if (i + 1) % self.trust_upd == 0:
                self.update_reputation_all()


class PeerTrustEnv(SimpleEnv):

    def __init__(self, num_peers, malicious_rate, min_cat_peer_rate, num_cats, trust_upd, collective=False):
        super().__init__(num_peers, malicious_rate, min_cat_peer_rate, num_cats, collective=collective)
        self.trust_upd = trust_upd
        self.transactions = []
        self.connected_peers = []
        self.reputation = np.zeros(self.num_peers)
        for i in range(num_peers):
            self.transactions.append([])
            self.connected_peers.append(set())

    def add_transaction(self, peer_i, peer_j, mark):
        """
        :param peer_i: from
        :param peer_j: to
        :param mark:
        :return:
        """
        self.transactions[peer_j].append((peer_i, mark))
        self.connected_peers[peer_j].add(peer_i)

    def choose_peers(self):
        buyer = np.random.choice(self.peers)
        cat = np.random.choice(list(buyer.cats))
        sellers, reps = [s for s in (self.cats[cat] - {buyer})], np.array([self.reputation[s.id] for s in
                                                                           (self.cats[cat] - {buyer})])
        with_zero = [s for s, r in zip(sellers, reps) if r == 0]

        # seller = sorted(sellers, key=lambda x: self.reputation[x.id], reverse=True)[0] #deterministic
        s = np.sum(reps)
        if len(with_zero) > 0:
            if np.random.rand() < 0.1 or s == 0:
                seller = np.random.choice(with_zero)
                return buyer, seller
        probs = [p / s for p in reps]
        seller = np.random.choice(sellers, p=probs)

        return buyer, seller

    def update_reputation(self):
        eps = 0.001
        t = np.ones(self.num_peers) * (1 / self.num_peers)
        t_next = np.ones(self.num_peers) * (1 / self.num_peers)
        dif = eps
        c = 0
        while dif >= eps:
            for v in range(self.num_peers):
                t_v = 0
                s = 0
                for i in self.connected_peers[v]:
                    s += t[i]
                if s != 0:
                    for tr in self.transactions[v]:
                        t_v += tr[1] * t[tr[0]] / s
                t_next[v] = t_v
            d = t_next - t
            dif = np.linalg.norm(d)
            t = t_next
            c += 1
        self.convergence.append(c)
        self.reputation = t_next

    def simulate(self, n_inter: int):
        np.random.seed(42)
        random.seed(42)
        for i in tqdm_notebook(range(n_inter)):
            # TODO: check choose peers
            buyer, seller = self.choose_peers()

            interaction = interact(buyer, seller)
            mark = 1 if interaction['mark'] else 0
            self.add_transaction(buyer.id, seller.id, mark)
            self.interactions.append(interaction)
            if (i + 1) % self.trust_upd == 0:
                self.update_reputation()


class PeerTrustFuzzyEnv(SimpleEnv):

    def __init__(self, num_peers, malicious_rate, min_cat_peer_rate, 
                 num_cats, trust_upd, defuzz_method='centroid', collective=False, fuzz_type=1):
        '''
        defuzz_method: str
            Controls which defuzzification method will be used.
                * 'centroid': Centroid of area
                * 'bisector': bisector of area
                * 'mom'     : mean of maximum
                * 'som'     : min of maximum
                * 'lom'     : max of maximum
        '''
        super().__init__(num_peers, malicious_rate, min_cat_peer_rate, num_cats, collective=collective)
        self.trust_upd = trust_upd
        self.transactions = []
        self.connected_peers = []
        self.defuzz_method = defuzz_method
        self.reputation = np.zeros(self.num_peers)
        self.fuzz_type = fuzz_type
        for i in range(num_peers):
            self.transactions.append([])
            self.connected_peers.append(set())
        if self.fuzz_type == 1:
            tr_v_bad = np.array([0., 0., 0.1, 0.2])
            tr_bad = np.array([0.1, 0.2, 0.4, 0.5])
            tr_norm = np.array([0.4, 0.5, 0.6, 0.7])
            tr_good = np.array([0.6, 0.7, 0.8, 0.9])
            tr_v_good = np.array([0.8, 0.9, 1., 1.])
        elif self.fuzz_type == 2:
            tr_v_bad = np.array([0., 0., 0., 0.1, 0.15, 0.25])
            tr_bad = np.array([0.05, 0.15, 0.2, 0.4, 0.45, 0.55])
            tr_norm = np.array([0.35, 0.45, 0.5, 0.6, 0.65, 0.75])
            tr_good = np.array([0.55, 0.65, 0.7, 0.8, 0.85, 0.95])
            tr_v_good = np.array([0.75, 0.85, 0.9, 1., 1., 1.])
        else:
            raise NotImplementedError
        self.mbfs = {'very bad': tr_v_bad, 'bad': tr_bad, 'normal': tr_norm, 'good': tr_good, 'very good': tr_v_good}

    def add_transaction(self, peer_i, peer_j, mark):
        """
        :param peer_i: from
        :param peer_j: to
        :param mark:
        :return:
        """
        self.transactions[peer_j].append((peer_i, self.mbfs[mark]))
        self.connected_peers[peer_j].add(peer_i)

    def choose_peers(self):
        buyer = np.random.choice(self.peers)
        cat = np.random.choice(list(buyer.cats))
        sellers, reps = [s for s in (self.cats[cat] - {buyer})], np.array([self.reputation[s.id] for s in
                                                                           (self.cats[cat] - {buyer})])
        with_zero = [s for s, r in zip(sellers, reps) if r == 0]

        # seller = sorted(sellers, key=lambda x: self.reputation[x.id], reverse=True)[0] #deterministic
        s = np.sum(reps)
        if len(with_zero) > 0:
            if np.random.rand() < 0.1 or s == 0:
                seller = np.random.choice(with_zero)
                return buyer, seller
        probs = [p / s for p in reps]
        seller = np.random.choice(sellers, p=probs)

        return buyer, seller

    def update_reputation(self):
        eps = 0.1
        t = np.ones(self.num_peers) * (1 / self.num_peers)
        t_next = np.ones(self.num_peers) * (1 / self.num_peers)
        dif = eps
        c = 0
        while dif >= eps:
            for v in range(self.num_peers):
                t_v = 0
                s = 0
                for i in self.connected_peers[v]:
                    s += t[i]
                if s != 0:
                    x = 1
                    fuzzy_t_v = self.transactions[v][0][1]
                    for tr in self.transactions[v][1:]:
                        crisp = t[tr[0]] / s
                        r = PeerTrustFuzzyEnv.fuzzy_mul(tr[1], crisp)
                        fuzzy_t_v = PeerTrustFuzzyEnv.fuzzy_add(fuzzy_t_v, r)
                    if self.fuzz_type == 1:
                        pass
                    elif self.fuzz_type == 2:
                        fuzzy_t_v = PeerTrustFuzzyEnv.defuzz_2(fuzzy_t_v)
                    else:
                        raise NotImplementedError
                    f = fuzzy_t_v, np.array([0, 1, 1, 0])
                    t_v = fuzz.defuzz(f[0], f[1], self.defuzz_method)
                t_next[v] = t_v
            d = t_next - t
            dif = np.linalg.norm(d)
            t = t_next
            c += 1
        self.convergence.append(c)
        self.reputation = t_next

    @staticmethod
    def fuzzy_add(l: np.ndarray, r: np.ndarray):
        return l + r

    @staticmethod
    def fuzzy_mul(f: np.ndarray, num: float):
        return f * num
    
    @staticmethod
    def defuzz_2(f: np.ndarray):
        l, r = np.mean([f[0], f[1]]), np.mean([f[4], f[5]])
        return np.array([l, f[2], f[3], r])

    def simulate(self, n_inter: int):
        np.random.seed(42)
        random.seed(42)
        for i in tqdm_notebook(range(n_inter)):
            # TODO: check choose peers
            buyer, seller = self.choose_peers()

            interaction = interact_fuzzy(buyer, seller)
            self.add_transaction(buyer.id, seller.id, interaction['mark'])
            self.interactions.append(interaction)
            if (i + 1) % self.trust_upd == 0:
                self.update_reputation()


class AbsoluteTrustEnv(SimpleEnv):

    def __init__(self, num_peers, malicious_rate, min_cat_peer_rate, num_cats, trust_upd, good_w=10, bad_w=1, p=3, q=1, collective=False):
        super().__init__(num_peers, malicious_rate, min_cat_peer_rate, num_cats, collective=collective)
        self.bad_w = bad_w
        self.good_w = good_w
        self.trust_upd = trust_upd
        self.reputation = np.ones(self.num_peers)
        self.marks = np.zeros((self.num_peers, self.num_peers, 2))
        self.local_trust_matrix = np.zeros((self.num_peers, self.num_peers))
        self.e = np.identity(self.num_peers)
        self.C = np.zeros((self.num_peers, self.num_peers))
        self.p = p
        self.q = q

    def choose_peers(self):
        buyer = np.random.choice(self.peers)
        cat = np.random.choice(list(buyer.cats))
        sellers, reps = [s for s in (self.cats[cat] - {buyer})], np.array([self.reputation[s.id] for s in
                                                                           (self.cats[cat] - {buyer})])
        with_zero = [s for s, r in zip(sellers, reps) if r == 0]

        # seller = sorted(sellers, key=lambda x: self.reputation[x.id], reverse=True)[0] #deterministic
        s = np.sum(reps)
        if len(with_zero) > 0:
            if np.random.rand() < 0.1 or s == 0:
                seller = np.random.choice(with_zero)
                return buyer, seller
        probs = [p / s for p in reps]
        seller = np.random.choice(sellers, p=probs)

        return buyer, seller

    def update_reputation(self):
        eps = 0.001

        t_next = np.zeros(self.num_peers)
        cs = np.zeros(self.num_peers)
        for k in range(self.num_peers):
            t = self.reputation.copy()
            dif = eps
            while dif >= eps:
                t_next[k] = self.t_k(t, k)
                dif = abs(t_next[k] - t[k])
                t[k] = t_next[k]
                cs[k] += 1
        self.convergence.append(cs.max())
        self.reputation = t_next

    def t_k(self, t, k):
        s1, s2, s3 = 0, 0, 0
        for j in range(self.num_peers):
            if self.local_trust_matrix[j, k] > 0:
                s1 += self.local_trust_matrix[j, k] * t[j]
                s2 += t[j]
                s3 += t[j] * t[j]
        if s2 == 0:
            return 0
        return ((s1 / s2) ** self.p * (s3 / s2) ** self.q) ** (1 / (self.p + self.q))

    def simulate(self, n_inter: int):
        np.random.seed(42)
        random.seed(42)
        for i in tqdm_notebook(range(n_inter)):
            buyer, seller = self.choose_peers()

            interaction = interact(buyer, seller)
            mark = 1 if interaction['mark'] else 0
            self.update_local_trust_matrix(buyer.id, seller.id, mark)
            self.interactions.append(interaction)
            if (i + 1) % self.trust_upd == 0:
                self.update_reputation()

    def update_local_trust_matrix(self, buyer_id, seller_id, mark):
        self.C[seller_id, buyer_id] = 1
        self.marks[buyer_id, seller_id, mark] += 1
        s = np.sum(self.marks[buyer_id, seller_id])
        x, y = self.marks[buyer_id, seller_id, 1] / s, self.marks[buyer_id, seller_id, 0] / s
        self.local_trust_matrix[buyer_id, seller_id] = 0.5 * ((x - y + 1) * self.good_w + (y - x + 1) * self.bad_w)


def interact(buyer: Peer, seller: Peer):
    info = buyer.buy(seller)
    return info

def count_stat(interactions: list):
    return sum([1 for x in interactions if not x['success']]) / len(interactions)

def interact_fuzzy(buyer: Peer, seller: Peer):
    info = buyer.buy_fuzzy(seller)
    return info