library(tidyverse)
library(car)

# %% --------
cf = read_csv('PureOptimal-term.csv')
z_score = function(x) {(x - mean(x)) / sd(x)}

cf = cf_raw %>% 
    filter(n_revealed < 16)  %>% 
    mutate(
        best_next_raw = best_next,
        term_reward = z_score(term_reward),
        max_path = z_score(max_path),
        max_competing = z_score(max_competing),
        best_next = z_score(best_next),
        potential_gain = z_score(potential_gain),
        n_revealed = z_score(n_revealed),
    )

# %% --------
full = glm(is_term ~ best_next + potential_gain + term_reward + n_revealed, family='binomial', data=cf)
summary(full)
anova(full)
Anova(full)

# %% --------
m2 = glm(is_term ~ best_next + potential_gain, family='binomial', data=cf)
Anova(m2)
summary(m2)

# %% --------
m1 = glm(is_term ~ term_reward * n_revealed, family='binomial', data=cf)
m2 = glm(is_term ~ best_next, family='binomial', data=cf)

anova(m1, m2, test='Rao')

summary(m2)

# %% --------
ggplot(cf_raw, aes(best_next, is_term)) + 
geom_smooth(method = "glm", 
    method.args = list(family = "binomial"), 
    se = FALSE) 
# %% --------
ggplot(cf_raw, aes(best_next, is_term)) + stat_summary(fun.data=mean_se)

# %% --------

cf %>% filter(best_next_raw == 10) %>% 
    glm(is_term ~ potential_gain + term_reward + n_revealed, family='binomial', data=.) %>% 
    summary


    # ggplot(aes(potential_gain, is_term)) + stat_summary(fun.data=mean_se)

# %% --------
summary(m1)

summary(update(m2, . ~ . + term_reward))
summary(update(m2, . ~ . + potential_gain))

summary(lm(m2r ~ n_revealed + term_reward + potential_gain, data=cf))

# %% --------
summary(glm(is_term ~ potential_gain, family='binomial', data=cf))
summary(glm(is_term ~ best_next, family='binomial', data=cf))

m1 = glm(is_term ~ term_reward + n_revealed, family='binomial', data=cf)
m2 = glm(is_term ~ term_reward * n_revealed, family='binomial', data=cf)
summary(m2)

anova(m1, m2)

anova(m2, test="Chisq")