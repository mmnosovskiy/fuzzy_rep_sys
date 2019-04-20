from environment import *
import matplotlib.pyplot as plt


def malicious_rate_exp(rates, sim_num, peer_num, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    res_dict = {'rates': rates, 'simple': [], 'eigen': [], 'honest': [], 'peer': [], 'abs': [], 'peer_eigen': [], 'peer_fuzzy': []}
    for rate in rates:
        t = []
        for i in range(exp_iter):
            simple = SimpleEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                               num_cats=cats_num)
            simple.simulate(sim_num)
            t.append(count_stat(simple.interactions))
        res_dict['simple'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            eigen = EigenTrustEnv(num_peers=peer_num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                                  min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            eigen.simulate(sim_num)
            t.append(count_stat(eigen.interactions))
        res_dict['eigen'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            honest = HonestPeerEnv(num_peers=peer_num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                                  min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            honest.simulate(sim_num)
            t.append(count_stat(honest.interactions))
        res_dict['honest'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer = PeerTrustEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd)
            peer.simulate(sim_num)
            t.append(count_stat(peer.interactions))
        res_dict['peer'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            absolute = AbsoluteTrustEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                        num_cats=cats_num, trust_upd=trust_upd)
            absolute.simulate(sim_num)
            t.append(count_stat(absolute.interactions))
        res_dict['abs'].append(np.mean(t))
        
        print(f'Rate {rate:.3f} completed!')

    return res_dict


def convergence_exp(nums, sim_num, rate, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    res_dict = {'nums': nums, 'simple': [], 'eigen': [], 'honest': [], 'peer': [], 'abs': [], 'peer_eigen': [], 'peer_fuzzy': []}
    for num in nums:
        t = []
        for i in range(exp_iter):
            eigen = EigenTrustEnv(num_peers=num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                              min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            eigen.simulate(sim_num)
            t.append(np.mean(eigen.convergence))
        res_dict['eigen'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            honest = HonestPeerEnv(num_peers=peer_num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                                  min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            honest.simulate(sim_num)
            t.append(np.mean(honest.convergence))
        res_dict['honest'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer = PeerTrustEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd)
            peer.simulate(sim_num)
            t.append(np.mean(peer.convergence))
        res_dict['peer'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            absolute = AbsoluteTrustEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                        num_cats=cats_num, trust_upd=trust_upd)
            absolute.simulate(sim_num)
            t.append(np.mean(absolute.convergence))
        res_dict['abs'].append(np.mean(t))
        
        print(f'Num {num} completed!')

    return res_dict


def robustness_exp(nums, sim_num, rate, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    res_dict = {'nums': nums, 'simple': [], 'eigen': [], 'honest': [], 'peer': [], 'abs': [], 'peer_eigen': [], 'peer_fuzzy': []}
    for num in nums:
        
        t = []
        for i in range(exp_iter):
            eigen = EigenTrustEnv(num_peers=num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                                  min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            eigen.simulate(sim_num)
            t.append(count_stat(eigen.interactions))
        res_dict['eigen'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            honest = HonestPeerEnv(num_peers=num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                                  min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            honest.simulate(sim_num)
            t.append(count_stat(honest.interactions))
        res_dict['honest'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer = PeerTrustEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd)
            peer.simulate(sim_num)
            t.append(count_stat(peer.interactions))
        res_dict['peer'].append(np.mean(t))

        t = []
        for i in range(exp_iter):
            absolute = AbsoluteTrustEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                        num_cats=cats_num, trust_upd=trust_upd)
            absolute.simulate(sim_num)
            t.append(count_stat(absolute.interactions))
        res_dict['abs'].append(np.mean(t))
        
        print(f'Num {num} completed!')

    return res_dict


def malicious_rate_exp_1(rates, sim_num, peer_num, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    res_dict = {'rates': rates, 'simple': [], 'eigen': [], 'honest': [], 'peer': [], 'abs': [], 'peer_eigen': [], 'peer_fuzzy': []}
    for rate in rates:
        t = []
        for i in range(exp_iter):
            simple = SimpleEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                               num_cats=cats_num)
            simple.simulate(sim_num)
            t.append(count_stat(simple.interactions))
        res_dict['simple'].append(np.mean(t))

        t = []
        for i in range(exp_iter):
            eigen = EigenTrustEnv(num_peers=peer_num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                                  min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            eigen.simulate(sim_num)
            t.append(count_stat(eigen.interactions))
        res_dict['eigen'].append(np.mean(t))

        t = []
        for i in range(exp_iter):
            peer = PeerTrustEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd)
            peer.simulate(sim_num)
            t.append(count_stat(peer.interactions))
        res_dict['peer'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['peer_fuzzy'].append(np.mean(t))
        
        print(f'Rate {rate:.3f} completed!')

    return res_dict


def convergence_exp_1(nums, sim_num, rate, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    res_dict = {'nums': nums, 'simple': [], 'eigen': [], 'honest': [], 'peer': [], 'abs': [], 'peer_eigen': [], 'peer_fuzzy': []}
    for num in nums:
        t = []
        for i in range(exp_iter):
            eigen = EigenTrustEnv(num_peers=num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                              min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            eigen.simulate(sim_num)
            t.append(np.mean(eigen.convergence))
        res_dict['eigen'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer = PeerTrustEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd)
            peer.simulate(sim_num)
            t.append(np.mean(peer.convergence))
        res_dict['peer'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(np.mean(peer_fuzzy.convergence))
        res_dict['peer_fuzzy'].append(np.mean(t))
        
        print(f'Num {num} completed!')

    return res_dict


def robustness_exp_1(nums, sim_num, rate, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    res_dict = {'nums': nums, 'simple': [], 'eigen': [], 'honest': [], 'peer': [], 'abs': [], 'peer_eigen': [], 'peer_fuzzy': []}
    for num in nums:
        
        t = []
        for i in range(exp_iter):
            eigen = EigenTrustEnv(num_peers=num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                                  min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            eigen.simulate(sim_num)
            t.append(count_stat(eigen.interactions))
        res_dict['eigen'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer = PeerTrustEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd)
            peer.simulate(sim_num)
            t.append(count_stat(peer.interactions))
        res_dict['peer'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(np.mean(peer_fuzzy.convergence))
        res_dict['peer_fuzzy'].append(np.mean(t))
        
        print(f'Num {num} completed!')

    return res_dict


def defuzz_exp(methods, sim_num, peer_num, rate, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    res = []
    for method in methods:
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd, defuzz_method=method)
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res.append((np.mean(t), np.std(t)))
        print(f'Method {method} completed!')

    return res

def defuzz_malicious_rate_exp(rates, sim_num, peer_num, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    res_dict = {'rates': rates, 'centroid': [], 'bisector': [], 'mom': [], 'som': [], 'lom': []}
    for rate in rates:
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                defuzz_method='centroid', num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['centroid'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                defuzz_method='bisector', num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['bisector'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                defuzz_method='mom', num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['mom'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                defuzz_method='som', num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['som'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                defuzz_method='lom', num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['lom'].append(np.mean(t))
        
        print(f'Rate {rate:.3f} completed!')

    return res_dict

def defuzz_robustness_exp(nums, sim_num, rate, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    res_dict = {'nums': nums, 'centroid': [], 'bisector': [], 'mom': [], 'som': [], 'lom': []}
    for num in nums:
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                defuzz_method='centroid', num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['centroid'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                defuzz_method='bisector', num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['bisector'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                defuzz_method='mom', num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['mom'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                defuzz_method='som', num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['som'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                defuzz_method='lom', num_cats=cats_num, trust_upd=trust_upd)
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['lom'].append(np.mean(t))
        
        print(f'Num {num} completed!')

    return res_dict

def malicious_rate_exp_2(rates, sim_num, peer_num, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    res_dict = {'rates': rates, 'simple': [], 'eigen': [], 'honest': [], 'peer': [], 'abs': [], 'peer_eigen': [], 'peer_fuzzy': [], 'peer_fuzzy_2': []}
    for rate in rates:
        t = []
        for i in range(exp_iter):
            simple = SimpleEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                               num_cats=cats_num)
            simple.simulate(sim_num)
            t.append(count_stat(simple.interactions))
        res_dict['simple'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            eigen = EigenTrustEnv(num_peers=peer_num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                                  min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            eigen.simulate(sim_num)
            t.append(count_stat(eigen.interactions))
        res_dict['eigen'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer = PeerTrustEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd)
            peer.simulate(sim_num)
            t.append(count_stat(peer.interactions))
        res_dict['peer'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd, fuzz_type=1, defuzz_method='som')
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['peer_fuzzy'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy_2 = PeerTrustFuzzyEnv(num_peers=peer_num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd, fuzz_type=2, defuzz_method='som')
            peer_fuzzy_2.simulate(sim_num)
            t.append(count_stat(peer_fuzzy_2.interactions))
        res_dict['peer_fuzzy_2'].append(np.mean(t))
        
        print(f'Rate {rate:.3f} completed!')

    return res_dict

def robustness_exp_2(nums, sim_num, rate, min_cat_peer_rate, cats_num, trust_upd, pre_trusted_rate, exp_iter=5):
    res_dict = {'nums': nums, 'simple': [], 'eigen': [], 'honest': [], 'peer': [], 'abs': [], 'peer_eigen': [], 'peer_fuzzy': [], 'peer_fuzzy_2': []}
    for num in nums:
        t = []
        for i in range(exp_iter):
            simple = SimpleEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                               num_cats=cats_num)
            simple.simulate(sim_num)
            t.append(count_stat(simple.interactions))
        res_dict['simple'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            eigen = EigenTrustEnv(num_peers=num, malicious_rate=rate, pre_trusted_rate=pre_trusted_rate,
                                  min_cat_peer_rate=min_cat_peer_rate, num_cats=cats_num, trust_upd=trust_upd)
            eigen.simulate(sim_num)
            t.append(count_stat(eigen.interactions))
        res_dict['eigen'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer = PeerTrustEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd)
            peer.simulate(sim_num)
            t.append(count_stat(peer.interactions))
        res_dict['peer'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy = PeerTrustFuzzyEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd, fuzz_type=1, defuzz_method='som')
            peer_fuzzy.simulate(sim_num)
            t.append(count_stat(peer_fuzzy.interactions))
        res_dict['peer_fuzzy'].append(np.mean(t))
        
        t = []
        for i in range(exp_iter):
            peer_fuzzy_2 = PeerTrustFuzzyEnv(num_peers=num, malicious_rate=rate, min_cat_peer_rate=min_cat_peer_rate,
                                num_cats=cats_num, trust_upd=trust_upd, fuzz_type=2, defuzz_method='som')
            peer_fuzzy_2.simulate(sim_num)
            t.append(count_stat(peer_fuzzy_2.interactions))
        res_dict['peer_fuzzy_2'].append(np.mean(t))
        
        print(f'Num {num} completed!')

    return res_dict

def plot_malicious_rate_exp(res_dict, save=False):
    plt.figure(figsize=(12,9))
    plt.plot(res_dict['rates'] * 100, res_dict['simple'], 'b-o', label='Simple')
    plt.plot(res_dict['rates'] * 100, res_dict['eigen'], 'C1-o', label='EigenTrust')
    plt.plot(res_dict['rates'] * 100, res_dict['honest'], 'g-o', label='HonestPeer')
    plt.plot(res_dict['rates'] * 100, res_dict['peer'], 'r-o', label='PeerTrust')
    plt.plot(res_dict['rates'] * 100, res_dict['abs'], 'm-o', label='AbsoluteTrust')
    plt.title('Robustness of reputation systems')
    plt.xlabel('% of malicious peers')
    plt.ylabel('Rate of unsuccessful transactions')
    plt.ylim(0, 0.4)
    plt.legend()
    if save:
        plt.savefig('robust_2_with_fuzzy.png', bbox_inches='tight')
    plt.show()

def plot_convergence_exp(res_dict, save=False):
    plt.figure(figsize=(12,9))
    plt.plot(res_dict['nums'], res_dict['eigen'], 'C1-o', label='EigenTrust')
    plt.plot(res_dict['nums'], res_dict['honest'], 'g-o', label='HonestPeer')
    plt.plot(res_dict['nums'], res_dict['peer'], 'r-o', label='PeerTrust')
    plt.plot(res_dict['nums'], res_dict['abs'], 'm-o', label='AbsoluteTrust')
    plt.title('Speed of convergence')
    plt.xlabel('# of peers')
    plt.ylabel('# of iterations')
    plt.ylim(1, 5)
    plt.legend()
    if save:
        plt.savefig('conv_2_with_fuzzy.png', bbox_inches='tight')
    plt.show()
    
def plot_robust_exp(res_dict, save=False):
    plt.figure(figsize=(12,9))
    plt.plot(res_dict['nums'], res_dict['eigen'], 'C1-o', label='EigenTrust')
    plt.plot(res_dict['nums'], res_dict['honest'], 'g-o', label='HonestPeer')
    plt.plot(res_dict['nums'], res_dict['peer'], 'r-o', label='PeerTrust')
    plt.plot(res_dict['nums'], res_dict['abs'], 'm-o', label='AbsoluteTrust')
    plt.title('Robustness of reputation systems')
    plt.xlabel('# of peers')
    plt.ylabel('Rate of unsuccessful transactions')
    plt.ylim(0, 0.16)
    plt.legend()
    if save:
        plt.savefig('robust_3_with_fuzzy.png', bbox_inches='tight')
    plt.show()
    
def plot_malicious_rate_exp_1(res_dict, save=False):
    plt.figure(figsize=(12,9))
    plt.plot(res_dict['rates'] * 100, res_dict['simple'], 'b-o', label='Simple')
    plt.plot(res_dict['rates'] * 100, res_dict['eigen'], 'C1-o', label='EigenTrust')
    plt.plot(res_dict['rates'] * 100, res_dict['peer'], 'r-o', label='PeerTrust')
    plt.plot(res_dict['rates'] * 100, res_dict['peer_fuzzy'], 'y-o', label='Fuzzy PeerTrust')
    plt.title('Robustness of reputation systems')
    plt.xlabel('% of malicious peers')
    plt.ylabel('Rate of unsuccessful transactions')
    plt.ylim(0, 0.4)
    plt.legend()
    if save:
        plt.savefig('robust_2_with_fuzzy.png', bbox_inches='tight')
    plt.show()
    
def plot_convergence_exp_1(res_dict, save=False):
    plt.figure(figsize=(12,9))
    plt.plot(res_dict['nums'], res_dict['eigen'], 'C1-o', label='EigenTrust')
    plt.plot(res_dict['nums'], res_dict['peer'], 'r-o', label='PeerTrust', alpha=0.3)
    plt.plot(res_dict['nums'], res_dict['peer_fuzzy'], 'y-o', label='Fuzzy PeerTrust')
    plt.title('Speed of convergence')
    plt.xlabel('# of peers')
    plt.ylabel('# of iterations')
    plt.ylim(1, 5)
    plt.legend()
    if save:
        plt.savefig('conv_2_with_fuzzy.png', bbox_inches='tight')
    plt.show()
    
def plot_robust_exp_1(res_dict, save=False):
    plt.figure(figsize=(12,9))
    plt.plot(res_dict['nums'], res_dict['eigen'], 'C1-o', label='EigenTrust')
    plt.plot(res_dict['nums'], res_dict['peer'], 'r-o', label='PeerTrust')
    plt.plot(res_dict['nums'], res_dict['peer_fuzzy'], 'y-o', label='Fuzzy PeerTrust')
    plt.title('Robustness of reputation systems')
    plt.xlabel('# of peers')
    plt.ylabel('Rate of unsuccessful transactions')
    plt.ylim(0, 0.16)
    plt.legend()
    if save:
        plt.savefig('robust_3_with_fuzzy.png', bbox_inches='tight')
    plt.show()
    
def plot_defuzz_exp(methods, res, save=False):
    perf = [x[0] for x in res]
    errs = [x[1] for x in res]
    y_pos = np.arange(len(methods))
    plt.figure(figsize=(12,9))
    plt.bar(y_pos, perf, yerr=errs, align='center', alpha=0.5)
    plt.title('Robustness of defuzzification methods')
    plt.xticks(y_pos, methods)
    plt.ylabel('Rate of unsuccessful transactions')
    if save:
        plt.savefig('defuzz_methods_1.png', bbox_inches='tight')
    plt.show()
    
def plot_defuzz_malicious_rate_exp(res_dict, save=False):
    plt.figure(figsize=(12,9))
    plt.plot(res_dict['rates'] * 100, res_dict['centroid'], 'b-o', label='centroid')
    plt.plot(res_dict['rates'] * 100, res_dict['bisector'], 'C1-o', label='bisector')
    plt.plot(res_dict['rates'] * 100, res_dict['mom'], 'g-o', label='mom')
    plt.plot(res_dict['rates'] * 100, res_dict['som'], 'r-o', label='som')
    plt.plot(res_dict['rates'] * 100, res_dict['lom'], 'm-o', label='lom')
    plt.title('Robustness of fuzzy reputation systems')
    plt.xlabel('% of malicious peers')
    plt.ylabel('Rate of unsuccessful transactions')
    plt.ylim(0, 0.4)
    plt.legend()
    if save:
        plt.savefig('robust_1_defuzz_methods.png', bbox_inches='tight')
    plt.show()
    
def plot_defuzz_robust_exp(res_dict, save=False):
    plt.figure(figsize=(12,9))
    plt.plot(res_dict['nums'], res_dict['centroid'], 'b-o', label='centroid')
    plt.plot(res_dict['nums'], res_dict['bisector'], 'C1-o', label='bisector')
    plt.plot(res_dict['nums'], res_dict['mom'], 'g-o', label='mom')
    plt.plot(res_dict['nums'], res_dict['som'], 'r-o', label='som')
    plt.plot(res_dict['nums'], res_dict['lom'], 'm-o', label='lom')
    plt.title('Robustness of fuzzy reputation systems')
    plt.xlabel('# of peers')
    plt.ylabel('Rate of unsuccessful transactions')
    plt.ylim(0, 0.16)
    plt.legend()
    if save:
        plt.savefig('robust_defuzz_methods.png', bbox_inches='tight')
    plt.show()

def plot_malicious_rate_exp_2(res_dict, save=False):
    plt.figure(figsize=(12,9))
    plt.plot(res_dict['rates'] * 100, res_dict['simple'], 'b-o', label='Simple')
    plt.plot(res_dict['rates'] * 100, res_dict['eigen'], 'C1-o', label='EigenTrust')
    plt.plot(res_dict['rates'] * 100, res_dict['peer'], 'r-o', label='PeerTrust')
    plt.plot(res_dict['rates'] * 100, res_dict['peer_fuzzy'], 'g-o', label='Fuzzy PeerTrust')
    plt.plot(res_dict['rates'] * 100, res_dict['peer_fuzzy_2'], 'm-o', label='Fuzzy Type-2 PeerTrust')
    plt.title('Robustness of reputation systems')
    plt.xlabel('% of malicious peers')
    plt.ylabel('Rate of unsuccessful transactions')
    plt.ylim(0, 0.4)
    plt.legend()
    if save:
        plt.savefig('robust_with_type_2_fuzzy.png', bbox_inches='tight')
    plt.show()
    
def plot_robust_exp_2(res_dict, save=False):
    plt.figure(figsize=(12,9))
    plt.plot(res_dict['nums'], res_dict['eigen'], 'C1-o', label='EigenTrust')
    plt.plot(res_dict['nums'], res_dict['peer'], 'r-o', label='PeerTrust')
    plt.plot(res_dict['nums'], res_dict['peer_fuzzy'], 'g-o', label='Fuzzy PeerTrust')
    plt.plot(res_dict['nums'], res_dict['peer_fuzzy_2'], 'm-o', label='Fuzzy Type-2 PeerTrust')
    plt.title('Robustness of reputation systems')
    plt.xlabel('# of peers')
    plt.ylabel('Rate of unsuccessful transactions')
    plt.ylim(0, 0.16)
    plt.legend()
    if save:
        plt.savefig('robust_exp_with_type_2_fuzzy.png', bbox_inches='tight')
    plt.show()