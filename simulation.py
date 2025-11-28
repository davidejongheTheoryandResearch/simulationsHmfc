"""
Dit programma simuleert 2 groepen (controle vs "OCD-achtige" groep) gebruik makend van dit model:
    y_t ~ Bernoulli( f( w^T u_t + x_t ) )

    met:
    - u_t : input vector op trial t
    - w   : gewichten op de inputs
    - x_t : latent criterium, AR(1)-proces
    - f   : logistische link (sigmoid)

- Onderliggende (ware) d' is gelijk in beide groepen.
- De OCD-groep heeft grotere criteriumfluctuaties (hogere standaardafwijking van error component (eps) en a dichter bij 1)
- we passen achteraf klassieke SDT met statisch criterium toe en kijken naar geobserveerde d' per subject
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
#--------------------------------------------------------------------------
# d' en criterium c berekenen van klassieke SDT taak (zonder fluctuations)
#--------------------------------------------------------------------------

def dprime_berekening(stim, resp):
    """
    stim: 0 = noise, 1 = signal
    resp: 0 = 'nee', 1 = 'ja'
    """
    stim = np.asarray(stim)
    resp = np.asarray(resp)

    #Hit rate: P(ja|signal)
    hit_rate = np.mean(resp[stim == 1] == 1)
    #False alarm rate: P(ja|noise)
    fa_rate = np.mean(resp[stim == 0] == 1)

    #voorkom 0 of 1 (anders z naar oneindig) 
    # heb ik moeten vragen aan AI
    eps = 1e-4
    hit_rate = np.clip(hit_rate, eps, 1-eps)
    fa_rate = np.clip(fa_rate, eps, 1-eps)

    zH = norm.ppf(hit_rate)
    zF = norm.ppf(fa_rate)

    d_prime = zH - zF
    c = -.5 * (zH + zF)

    return d_prime, c

# ------------------------------------------------
# functie sigmoid transformatie
# ------------------------------------------------

def logistic(z):
    return 1.0/(1.0 + np.exp(-z))

# ------------------------------------------------------------
# 1 subject simuleren: y_t ~ Bernoulli( f( w^T u_t + x_t ) )
# ------------------------------------------------------------

def simulatie_bernoulli(
        n_trials = 1000,
        n_inputs = 3,
        a = .9,
        sigma_eps = .3,
        b = 0,
        w = None,
        p_signal = .5,
        d_true = 1.5, 
        seed = None
):
    """
    Simuleert één subject met het model:
        x_t = b + a * x_{t-1} + eps_t      (AR(1) criterium)
        z_t = w^T u_t + x_t                (lineaire voorspeller)
        p_t = logistic(z_t)
        y_t ~ Bernoulli(p_t)

    u_t bevat hier o.a. een stimulus-dimensie:
    - u[:, 0] = stim (0 = noise, 1 = signal)
    - u[:, 1:] zijn de andere covariaten (ruis), mean-centered
    Voor dit trucje heb ik AI even geraadpleegd

    PARAMETERS:
        n_trials = aantal trials
        n_inputs = dimensie van u_t
        a = autocorrelatie van AR(1)
        sigma_eps = standaardafwijking van de ruis eps_t
        b = Intercept van AR(1)
        w = vector met gewichten (als None => sample uit N(0, 0.5^2))
        p_signal = kans op een signal-trial
        Seed = Random seed (handig om te reproduceren)
    """

    rng = np.random.default_rng(seed)

    # Stimulus: 0 = noise, 1 = signal
    stim = rng.binomial(1, p_signal, size = n_trials)

    # Gewichten w:
    if w is None:
        w = np.zeros(n_inputs)
        w[0] = d_true #dit zorgt ervoor dat beide groepen dezelfde ECHTE gevoeligheid hebben

    # Inputs u: eerste kolom = stim, rest is ruis
    u = np.zeros((n_trials, n_inputs))
    u[:, 0] = stim

    if n_inputs > 1:
        other = rng.normal(loc = 0.0, scale = 1.0, size = (n_trials, n_inputs-1))
        other -= other.mean(axis = 0, keepdims = True) #mean is 0
        u[:, 1:] = other

    # Latent criterium x_t 
    x = np.zeros(n_trials) #We initiëren een array van lengte n_trials waarin elke waarde gelijk is aan 0
    x[0] = 0.0
    for t in range(1, n_trials):
        eps_t = rng.normal(loc = 0.0, scale = sigma_eps)
        x[t] = b + a * x[t-1] + eps_t

    #Lineaire voorspeller en kans op respons (AI gebruikt voor dit deeltje)
    # z_t = w^T u_t + x_t
    dv = u @ w #decision variable
    linpred = dv + x
    p = logistic(linpred)

    #Binaire respons y_t ~ bernoulli(p_t)
    y = rng.binomial(1, p, size = n_trials)

    #confidence: |decision variable - criterium|
    conf = np.abs(dv - x)

    return {
        "y" : y,
        "p" : p,
        "x" : x,
        "u" : u,
        "w" : w,
        "stim" : stim,
        "conf" : conf
    }

#------------------------------------------------
# Twee groepen simuleren (controle vs "OCD")
#------------------------------------------------

def simulate_group(
    n_subj = 30,
    n_trials = 1000,
    n_inputs = 3,
    a = .9,
    sigma_eps = .3,
    b = 0.0,
    d_true = 1.5,
    seed = 0
):
    """
    Simuleert een groep mensen met dezelfde a, sigma_eps, b.
    Maar elke persoon heeft andere w en u
    """
    rng = np.random.default_rng(seed)
    subjects = []

    for i in range(n_subj):
        subj_seed = rng.integers(1e9) #zorgt voor willekeur
        sim = simulatie_bernoulli(
            n_trials= n_trials,
            n_inputs= n_inputs,
            a = a,
            sigma_eps= sigma_eps,
            b= b,
            w = None,
            p_signal= .5,
            d_true = d_true,
            seed = subj_seed
        )
        subjects.append(sim)
    return subjects

def main():
    # Parameters
    n_subj = 300
    n_trials = 5000
    n_inputs = 3
    d_true = 1.5

    # Controlegroep: lagere fluctuaties
    a_controle = .5
    sigma_controle = .15

    # OCD-groep: hogere fluctuaties
    a_OCD = .95
    sigma_OCD = .40

    #simuleer de groepen
    controle_subjects = simulate_group(
        n_subj= n_subj,
        n_trials= n_trials,
        n_inputs= n_inputs,
        a = a_controle,
        sigma_eps= sigma_controle,
        b = 0,
        d_true= d_true,
        seed = None
    )

    OCD_subjects = simulate_group(
        n_subj= n_subj,
        n_trials= n_trials,
        n_inputs= n_inputs,
        a = a_OCD,
        sigma_eps= sigma_OCD,
        b = 0,
        d_true= d_true,
        seed = None
    )

    #-----------------------------------
    # d' per subject berekenen
    #-----------------------------------

    d_controle = []
    d_OCD = []
    conf_mean_controle = []
    conf_mean_OCD = []

    for subj in controle_subjects:
        stim = subj["stim"]
        resp = subj["y"]
        d_prime, c = dprime_berekening(stim, resp)
        d_controle.append(d_prime)
        conf_mean_controle.append(subj["conf"].mean())

    for subj in OCD_subjects:
        stim = subj["stim"]
        resp = subj["y"]
        d_prime, c = dprime_berekening(stim, resp)
        d_OCD.append(d_prime)
        conf_mean_OCD.append(subj["conf"].mean())

    d_controle = np.array(d_controle)
    d_OCD = np.array(d_OCD)
    conf_mean_controle = np.array(conf_mean_controle)
    conf_mean_OCD      = np.array(conf_mean_OCD)

    print("--- Geobserveerde d' per groep (klassieke SDT-fit) ---")
    print(f"Controle: mean d' = {d_controle.mean():.3f}, SD = {d_controle.std():.3f}")
    print(f"OCD:      mean d' = {d_OCD.mean():.3f}, SD = {d_OCD.std():.3f}")

    print("--- Gemiddelde confidence per groep ---")
    print(f"Controle: mean conf = {conf_mean_controle.mean():.3f}, SD = {conf_mean_controle.std():.3f}")
    print(f"OCD:      mean conf = {conf_mean_OCD.mean():.3f}, SD = {conf_mean_OCD.std():.3f}")

    #-----------------------------------------------------------------------------------------------------------------
    # Grafische voorstelling (hier heb ik wel wat hulp gehad van AI om te weten te komen hoe je dit mooi vorm geeft)
    #-----------------------------------------------------------------------------------------------------------------

    # Gemiddelde en standaardfout per groep
    mean_d_controle = d_controle.mean()
    mean_d_OCD  = d_OCD.mean()
    mean_conf_controle = conf_mean_controle.mean()
    mean_conf_OCD = conf_mean_OCD.mean()

    se_d_controle = d_controle.std(ddof=1) / np.sqrt(len(d_controle))
    se_d_OCD  = d_OCD.std(ddof=1) / np.sqrt(len(d_OCD))
    se_conf_controle = conf_mean_controle.std(ddof=1) / np.sqrt(len(conf_mean_controle))
    se_conf_OCD = conf_mean_OCD.std(ddof=1) / np.sqrt(len(conf_mean_OCD))

    # t-test tussen groepen (AI geraadpleegd voor dit stukje)
    tval_dprime, pval_dprime = ttest_ind(d_controle, d_OCD, equal_var=False)
    tval_conf, pval_conf = ttest_ind(conf_mean_controle, conf_mean_OCD, equal_var=False)

    print(f"d': t = {tval_dprime:.3f}, p = {pval_dprime:.3e}")
    print(f"Confidence: t = {tval_conf:.3f}, p = {pval_conf:.3e}")

    #data in arrays
    means_controle = np.array([mean_d_controle, mean_conf_controle])
    means_OCD = np.array([mean_d_OCD, mean_conf_OCD])
    ses_controle = np.array([se_d_controle, se_conf_controle])
    ses_OCD = np.array([se_d_OCD, se_conf_OCD])

    labels = ["d'", "confidence"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(x - width/2, means_controle, width,
        yerr=ses_controle, capsize=5, label="Controle")
    ax.bar(x + width/2, means_OCD,  width,
        yerr=ses_OCD,  capsize=5, label="OCD")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gemiddelde waarde")
    ax.set_title("Groepsverschillen in d' en confidence (klassieke SDT)")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()

    # Kleine indicatie van significantie boven elke maat apart
    def p_als_sterren(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "n.s."

    # d'-significantie boven eerste paar balken
    y_max_d = max(means_controle[0] + ses_controle[0],
                  means_OCD[0]  + ses_OCD[0])
    y_sig_d = y_max_d + 0.05
    ax.plot([x[0] - width/2, x[0] + width/2],
            [y_sig_d, y_sig_d], color="black")
    ax.text(x[0], y_sig_d + 0.02, p_als_sterren(pval_dprime),
            ha="center", va="bottom")

    # confidence-significantie boven tweede paar balken
    y_max_c = max(means_controle[1] + ses_controle[1],
                  means_OCD[1]  + ses_OCD[1])
    y_sig_c = y_max_c + 0.05
    ax.plot([x[1] - width/2, x[1] + width/2],
            [y_sig_c, y_sig_c], color="black")
    ax.text(x[1], y_sig_c + 0.02, p_als_sterren(pval_conf),
            ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()