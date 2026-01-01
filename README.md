<p align="center">
  <a href="https://go-skill-icons.vercel.app/">
    <img
      src="https://go-skill-icons.vercel.app/api/icons?i=rust"
    />
  </a>
</p>

# Momentum anidation models.

This experiment was inspired by my (so far partial) lecture on [Nested Learning: The Illusion of Deep Learning Architecture](https://abehrouz.github.io/files/NL.pdf).

The basic idea (at least what I understood) to motivate this experiment is that the Momentum optimizer mechanic is a long-term memory mechanism.

## The experiment

My idea was to prove this over a dataset of random-generated in a regression task where the real weights were changing over time.

Both the weights of the model and the real weights of the function ( $y = w \cdot x + \epsilon$, $\epsilon \sim N(0,2)$ ), have several levels of nested momentum. 

The idea of each momentum level $M_i$ is that $M_m$ is the gradient in the case of the model backward step, and a random variable in the case of the real weights change. Then, each $M_i$ ($i < m$) is updated to $M_i' = \alpha \cdot M_i + (1-\alpha) \cdot M_{i+1}$, with $\alpha = 0.95$.

The only difference between model and real weights is that the last step, in the real weights uses the same formula, while in the model it uses a normal gradient descend step $w' = w - lr \cdot M_0$.

## The hypothesis

If they use the same underlying coefficients, I expected that the model loss in the online regression task (30.000 samples, first 1.000 are discarded as warm-up) was minimized when both model and reality have the same numer of levels of nested momentum.

## The results

Against what I expected, it happened to be that increasing the number of levels in the model increased the lost and increasing the number of levels in the real weights decreased the loss.

Even more, the effect in the model levels is exponential.

You can see it in the results (you can recreate them using the script, due the fact that the seed is fixed in the code)

(CI at 95%)

```
=== Linear Regression: loss ~ model_level + data_level ===
n = 361, R² = 0.1210

Parameter           Estimate      Std.Err     CI Lower     CI Upper Pr(>|t|)
---------------------------------------------------------------------------
(Intercept)     -7593125971676635583195990260064853038928637928344026175638442375298750087168.0000  3142347707867816203403235205501000349869481868846203350384723208644472078336.0000 -13772906312533247529369653998454691742781082759037243087498423840157210771456.0000 -1413345630820024440491348651170152106057239268232110524879957801836707053568.0000   0.0162 *
model_level     1451237950856866179203317569417929992207816350078805812845585467422780948480.0000 207200271918800880901184638520482241374216495899687896348922454408783986688.0000 1043755302193272685296770595711748837276356798668905984520813119279604957184.0000 1858720599520459873977120075497895589884537444134030956445732038415061352448.0000   0.0000 ***
data_level      97545084439684404029962707334304167553874560296321939187565520585905668096.0000 207200271918800730250742989240143909315270338915693909892391787271955677184.0000 -309937564223908826238311380131284906274429216391588412838278160067820781568.0000 505027733103277609189829853253170186039020644153566626803987423383494066176.0000   0.6381
---------------------------------------------------------------------------
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1


=== Linear Regression: log(loss) ~ model_level + data_level ===
n = 361, R² = 0.8394

Parameter           Estimate      Std.Err     CI Lower     CI Upper Pr(>|t|)
---------------------------------------------------------------------------
(Intercept)         -23.4943       4.5257     -32.3946     -14.5940   0.0000 ***
model_level          12.9085       0.2984      12.3216      13.4954   0.0000 ***
data_level           -0.0347       0.2984      -0.6216       0.5522   0.9075
---------------------------------------------------------------------------
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## My conjetures after the results

- This naive nested momentums generate some kind of amplification or numerical problems with the model weights.
- Nesting this way vanish the impact of new gradients, avoiden the model to adapt to recent data. It's like it's too biased, and in a divergent way.

```
Disclaimer: The code, in particular the crate's specific use, was assisted by IA. I do not find it importat, but some people do. 
```
