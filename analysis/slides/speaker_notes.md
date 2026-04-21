# Speaker Notes: Mixed-Effects Model Presentation

**Presentation Context:** First mockup of park-specific weather effects model
**Audience:** Analytics team with basic statistics background
**Time:** ~1 minute per slide

---

## Slide 1: Data Pipeline
*"Isolating Park-Specific Weather Effects on Game Outcomes"*

### Key Talking Points

**The Question:**
"We're trying to answer: does weather affect strikeouts differently at Oracle Park versus Coors Field versus Fenway? Not just 'does weather matter' — but 'does it matter *differently* depending on where you play?'"

**Data Overview:**
"We pulled game-level data from all 30 MLB parks — about 10,000 games from 2015 to 2024. Each game has weather readings: temperature, humidity, wind speed, and wind direction broken into components toward center field."

**Why Away Team Performance:**
"We're predicting *away team* strikeouts and runs, not home team. This is intentional — it helps us isolate the park effect from home-field advantage. The away team rotates through, so we're measuring what the *park* does to a neutral set of hitters."

**What This Model Does:**
- Estimates average weather effects across the league
- Identifies which parks produce more/fewer strikeouts than expected
- Controls for the fact that some teams just strike out more than others

**What This Model Does NOT Do:**
- Account for specific pitcher-batter matchups
- Distinguish between good offensive teams vs. weak ones visiting
- Use real-time in-stadium weather (we use nearby weather stations)

---

## Slide 2: Model Hierarchy
*"Building Complexity: 4-Model Hierarchy"*

### Key Talking Points

**Why Not Just Linear Regression:**
"A simple regression would say 'temperature has X effect on strikeouts' — but that assumes the effect is the same everywhere. We know that's not true. A 10-degree temperature swing at Coors Field probably matters differently than at a dome."

**The Hierarchy Explained:**

1. **Null Model:** "This is our baseline — just captures that some teams strike out more than others and that league-wide K-rates change year to year. No weather yet."

2. **Fixed Weather:** "Now we add weather, but we're estimating a single, league-wide effect. 'On average, a 1-SD increase in temperature is associated with X fewer strikeouts.'"

3. **Park Intercept:** "This is where it gets interesting. We let each park have its own baseline — Coors can be systematically lower, Houston can be systematically higher. This captures altitude, park dimensions, local climate."

4. **Park Slopes:** "The most complex model — now each park can have its own *response* to weather. Maybe temperature matters more at outdoor parks than domes."

**Model Selection:**
"We use AIC to compare models. Lower is better. Each step down shows improvement, meaning these additional complexities are justified by the data — they're not just overfitting."

**Key Limitation:**
"This model treats all away teams as equivalent. It doesn't know that the Dodgers have a better lineup than the A's. That variance gets absorbed into the 'residual' bucket."

---

## Slide 3: Variance Decomposition
*"Where Does the Variance Come From?"*

### Key Talking Points

**The Sobering Reality:**
"First thing to notice: almost 90% of the variance is unexplained. Game outcomes are noisy. This isn't a failure of the model — it's the reality of baseball. A single game has huge randomness."

**What We Can Explain:**
- **Parks (~5%):** "After controlling for who's batting, parks still differ. Coors has fewer strikeouts, Houston has more. This is real and consistent."
- **Teams (~1.5%):** "Some teams just strike out more — this is the away team's batting tendencies."
- **Season (~4%):** "League-wide K-rates have trended up over the decade. This captures that."

**The R² Story:**
"Marginal R² is about 8% — that's what weather alone explains. Conditional R² is about 18% — that's weather plus all the random effects. The 10-point gap shows why controlling for team and park matters."

**Confounders Worth Considering (Future Work):**

1. **Pitcher quality:** "We're not controlling for who's on the mound. A deGrom start vs. a spot starter is huge."

2. **Lineup strength:** "The model treats the 2023 Dodgers lineup the same as the 2023 A's. That's a problem."

3. **Roof status:** "For retractable roof stadiums (Houston, Miami, Arizona), we're not distinguishing open vs. closed."

4. **Game context:** "Day games after night games, doubleheaders, September callups — none of this is modeled."

**Bottom Line:**
"This is a first mockup. The park effects are real and worth investigating. But for game-day predictions, we'd need to layer in pitcher/batter matchup data. Right now, this is better for understanding *trends* than making sharp predictions."

---

## General Q&A Prep

**"Can we use this to predict tonight's game?"**
> "Not reliably yet. The model captures real effects, but 90% unexplained variance means any single-game prediction has wide error bars. It's better for understanding systematic patterns — like 'expect fewer strikeouts at Coors' — than predicting exact totals."

**"Why not include pitcher ERA or batter stats?"**
> "That's the logical next step. This version isolates weather and park effects. Adding pitcher/batter features would improve predictions but also make it harder to interpret the 'pure' weather effect."

**"Is the temperature effect causal?"**
> "We can't say for certain. Temperature correlates with other things — time of year, game time, crowd size. The effect is robust after controlling for what we control for, but hidden confounders may exist."

**"Which parks should we care about most?"**
> "Coors is the outlier — nearly 1 full strikeout below average. Houston and the Mets' park are on the high end. Oracle Park (SF) is slightly below average, consistent with being a pitcher's park."
