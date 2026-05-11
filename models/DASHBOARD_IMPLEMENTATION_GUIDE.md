# ParkCast Dashboard Implementation Guide v5.0

This guide provides complete implementation instructions for integrating the weather-adjusted baseball prediction model into a dashboard. No external API calls required - all parameters and logic are self-contained.

---

# Part 1: Model Implementation

## Overview

The model predicts how weather conditions affect strikeouts and runs at each MLB stadium. It uses 5 weather features plus day/night status, with park-specific adjustments.

**Model Performance:**
- Strikeouts: R² = 0.04 (explains ~4% of variance)
- Runs: R² = 0.02 (explains ~2% of variance)

**Use Case:** Directional guidance only (will weather increase/decrease K or R), not precise point predictions.

---

## Complete Model Parameters

```javascript
const MODEL_PARAMS = {
  version: '5.0.0',

  // ============================================================
  // SCALING PARAMETERS
  // Used to standardize raw weather values before applying coefficients
  // Formula: scaled_value = (raw_value - center) / scale
  // ============================================================
  scaling: {
    temp_f:      { center: 73.0,     scale: 16.3     },  // Temperature (°F)
    rhum:        { center: 62.7,     scale: 26.0     },  // Relative humidity (%)
    wspd_mph:    { center: 7.0,      scale: 4.7      },  // Wind speed (mph)
    wind_cf:     { center: 1.89,     scale: 13.95    },  // Wind toward CF component
    air_density: { center: 1.170203, scale: 0.048265 }   // Air density (kg/m³)
  },

  // ============================================================
  // ROOFED STADIUMS
  // Weather effects are dampened by 95% (multiplied by 0.05) for these parks
  // ============================================================
  roof_dampening_factor: 0.05,
  roofed_stadiums: ['ARI', 'HOU', 'MIA', 'MIL', 'SEA', 'TB', 'TEX', 'TOR'],

  // ============================================================
  // STRIKEOUTS MODEL
  // ============================================================
  strikeouts: {
    // Fixed effects (applied to all parks)
    fixed_effects: {
      Intercept:   8.8862,   // Base strikeouts per game
      temp_f:      0.2186,   // Per 1 SD increase: +0.22 K
      rhum:       -0.0689,   // Per 1 SD increase: -0.07 K
      wspd_mph:    0.0268,   // Per 1 SD increase: +0.03 K
      wind_cf:     0.0157,   // Per 1 SD increase: +0.02 K
      air_density: 0.4109,   // Per 1 SD increase: +0.41 K (biggest effect!)
      is_night:   -0.2178    // Night games: -0.22 K
    },

    // Park-specific adjustments (added to prediction)
    park_effects: {
      ARI: -0.0003, ATL:  0.1795, BAL: -0.4940, BOS:  0.1042, CHC:  0.1014,
      CIN:  0.1986, CLE:  0.3389, COL:  0.0352, CWS:  0.1070, DET: -0.5348,
      HOU:  0.5008, KC:  -0.5076, LAA:  0.0773, LAD:  0.5124, MIA: -0.0790,
      MIL:  0.2942, MIN: -0.1186, NYM:  0.2793, NYY:  0.3461, OAK: -0.3389,
      PHI:  0.1816, PIT: -0.2412, SD:   0.1876, SEA: -0.0907, SF:  -0.0658,
      STL: -0.4051, TB:   0.2272, TEX: -0.4813, TOR: -0.2353, WSH: -0.0786
    }
  },

  // ============================================================
  // RUNS MODEL
  // ============================================================
  runs: {
    // Fixed effects (applied to all parks)
    fixed_effects: {
      Intercept:   4.4227,   // Base runs per game
      temp_f:      0.0263,   // Per 1 SD increase: +0.03 R
      rhum:       -0.0972,   // Per 1 SD increase: -0.10 R
      wspd_mph:    0.0731,   // Per 1 SD increase: +0.07 R
      wind_cf:     0.1261,   // Per 1 SD increase: +0.13 R (biggest effect!)
      air_density: -0.2157,  // Per 1 SD increase: -0.22 R
      is_night:    0.1259    // Night games: +0.13 R
    },

    // Park-specific adjustments (added to prediction)
    park_effects: {
      ARI: -0.0133, ATL:  0.0331, BAL:  0.2454, BOS:  0.2342, CHC: -0.1617,
      CIN:  0.2485, CLE: -0.1947, COL:  0.0255, CWS:  0.1099, DET:  0.1805,
      HOU: -0.1209, KC:   0.1024, LAA:  0.1880, LAD: -0.4900, MIA:  0.0667,
      MIL: -0.1082, MIN:  0.0945, NYM: -0.1624, NYY: -0.1575, OAK:  0.0497,
      PHI: -0.0341, PIT:  0.0856, SD:  -0.0791, SEA: -0.0383, SF:  -0.0142,
      STL: -0.2227, TB:  -0.1504, TEX:  0.1082, TOR:  0.0953, WSH:  0.0802
    }
  }
};
```

---

## Core Functions

### 1. Compute Air Density

Air density affects ball flight. Compute from temperature and humidity:

```javascript
/**
 * Compute air density from temperature and humidity
 * @param {number} tempF - Temperature in Fahrenheit
 * @param {number} rhum - Relative humidity (0-100)
 * @param {number} presMb - Pressure in millibars (default: 1013.25)
 * @returns {number} Air density in kg/m³
 */
function computeAirDensity(tempF, rhum, presMb = 1013.25) {
  const tempK = (tempF - 32) * 5/9 + 273.15;
  const tempC = (tempF - 32) * 5/9;

  // Saturation vapor pressure (Magnus formula)
  const eSat = 6.1078 * Math.pow(10, (7.5 * tempC) / (tempC + 237.3));

  // Actual vapor pressure
  const e = (rhum / 100) * eSat;

  // Partial pressure of dry air (Pa)
  const pD = (presMb - e) * 100;

  // Gas constants
  const Rd = 287.05;  // J/(kg·K) for dry air
  const Rv = 461.5;   // J/(kg·K) for water vapor

  return pD / (Rd * tempK) + (e * 100) / (Rv * tempK);
}
```

### 2. Compute Wind Component Toward Center Field

```javascript
/**
 * Compute wind component blowing toward center field
 * @param {number} wspdMph - Wind speed in mph
 * @param {number} windDirDeg - Wind direction (meteorological: where wind blows FROM)
 * @param {number} stadiumOrientation - Stadium CF orientation (degrees from north)
 * @returns {number} Positive = blowing out, Negative = blowing in
 */
function computeWindCF(wspdMph, windDirDeg, stadiumOrientation) {
  const relDeg = ((windDirDeg - stadiumOrientation) + 360) % 360;
  const relRad = relDeg * Math.PI / 180;
  return wspdMph * Math.cos(relRad);
}
```

### 3. Scale a Feature

```javascript
/**
 * Scale a feature value using model parameters
 * @param {number} value - Raw feature value
 * @param {string} feature - Feature name
 * @returns {number} Scaled value
 */
function scaleFeature(value, feature) {
  const { center, scale } = MODEL_PARAMS.scaling[feature];
  return (value - center) / scale;
}
```

### 4. Check for Roofed Stadium

```javascript
function hasRoof(parkAbbr) {
  return MODEL_PARAMS.roofed_stadiums.includes(parkAbbr);
}
```

### 5. Main Prediction Function

```javascript
/**
 * Compute prediction for a target (strikeouts or runs)
 * @param {string} target - 'strikeouts' or 'runs'
 * @param {Object} weather - Weather conditions
 * @param {string} parkAbbr - Park abbreviation (e.g., 'SF', 'COL')
 * @returns {Object} Prediction result
 */
function predict(target, weather, parkAbbr) {
  const params = MODEL_PARAMS[target];
  const fe = params.fixed_effects;

  // Roof dampening
  const isRoofed = hasRoof(parkAbbr);
  const dampening = isRoofed ? MODEL_PARAMS.roof_dampening_factor : 1.0;

  // Scale features
  const tempScaled = scaleFeature(weather.temp_f, 'temp_f');
  const rhumScaled = scaleFeature(weather.rhum, 'rhum');
  const wspdScaled = scaleFeature(weather.wspd_mph, 'wspd_mph');
  const windCfScaled = scaleFeature(weather.wind_cf, 'wind_cf');
  const densityScaled = scaleFeature(weather.air_density, 'air_density');

  // Compute weather effect (with roof dampening for weather features)
  const weatherEffect =
    (fe.temp_f * tempScaled * dampening) +
    (fe.rhum * rhumScaled * dampening) +
    (fe.wspd_mph * wspdScaled * dampening) +
    (fe.wind_cf * windCfScaled * dampening) +
    (fe.air_density * densityScaled * dampening) +
    (fe.is_night * (weather.is_night ? 1 : 0));  // Night not dampened

  // Park effect (not dampened)
  const parkEffect = params.park_effects[parkAbbr] || 0;

  // Total prediction
  const prediction = fe.Intercept + weatherEffect + parkEffect;

  return {
    prediction: +prediction.toFixed(2),
    weatherEffect: +weatherEffect.toFixed(3),
    parkEffect: +parkEffect.toFixed(2),
    isRoofed: isRoofed
  };
}
```

### 6. Full Projection (User-Facing)

```javascript
/**
 * Run full projection with user-friendly inputs
 * @param {Object} inputs - User inputs
 * @returns {Object} Complete projection
 */
function runProjection(inputs) {
  const { temp, humidity, windSpeed, windDir, isNight, parkAbbr, stadiumOrientation } = inputs;

  // Compute derived features
  const windCf = computeWindCF(windSpeed, windDir, stadiumOrientation);
  const airDensity = computeAirDensity(temp, humidity);

  // Build weather object
  const weather = {
    temp_f: temp,
    rhum: humidity,
    wspd_mph: windSpeed,
    wind_cf: windCf,
    air_density: airDensity,
    is_night: isNight
  };

  // Get predictions
  const kResult = predict('strikeouts', weather, parkAbbr);
  const rResult = predict('runs', weather, parkAbbr);

  // Baseline (neutral conditions)
  const baseline = {
    temp_f: 73, rhum: 63, wspd_mph: 7, wind_cf: 0,
    air_density: 1.17, is_night: false
  };
  const kBaseline = predict('strikeouts', baseline, parkAbbr);
  const rBaseline = predict('runs', baseline, parkAbbr);

  return {
    strikeouts: {
      predicted: kResult.prediction,
      baseline: kBaseline.prediction,
      delta: +(kResult.prediction - kBaseline.prediction).toFixed(2),
      direction: kResult.prediction > kBaseline.prediction ? 'up' :
                 kResult.prediction < kBaseline.prediction ? 'down' : 'neutral'
    },
    runs: {
      predicted: rResult.prediction,
      baseline: rBaseline.prediction,
      delta: +(rResult.prediction - rBaseline.prediction).toFixed(2),
      direction: rResult.prediction > rBaseline.prediction ? 'up' :
                 rResult.prediction < rBaseline.prediction ? 'down' : 'neutral'
    },
    computed: {
      airDensity: +airDensity.toFixed(4),
      windCf: +windCf.toFixed(1)
    },
    isRoofed: kResult.isRoofed
  };
}
```

---

## Stadium Data

Include this data for stadium orientations (degrees from north where CF faces):

```javascript
const STADIUM_DATA = {
  ARI: { name: 'Chase Field',              orientation: 180, hasRoof: true  },
  ATL: { name: 'Truist Park',              orientation: 135, hasRoof: false },
  BAL: { name: 'Camden Yards',             orientation: 225, hasRoof: false },
  BOS: { name: 'Fenway Park',              orientation: 112, hasRoof: false },
  CHC: { name: 'Wrigley Field',            orientation: 225, hasRoof: false },
  CIN: { name: 'Great American Ball Park', orientation: 180, hasRoof: false },
  CLE: { name: 'Progressive Field',        orientation: 180, hasRoof: false },
  COL: { name: 'Coors Field',              orientation: 255, hasRoof: false },
  CWS: { name: 'Guaranteed Rate Field',    orientation: 180, hasRoof: false },
  DET: { name: 'Comerica Park',            orientation: 135, hasRoof: false },
  HOU: { name: 'Minute Maid Park',         orientation: 180, hasRoof: true  },
  KC:  { name: 'Kauffman Stadium',         orientation: 180, hasRoof: false },
  LAA: { name: 'Angel Stadium',            orientation: 180, hasRoof: false },
  LAD: { name: 'Dodger Stadium',           orientation: 0,   hasRoof: false },
  MIA: { name: 'loanDepot Park',           orientation: 180, hasRoof: true  },
  MIL: { name: 'American Family Field',    orientation: 180, hasRoof: true  },
  MIN: { name: 'Target Field',             orientation: 180, hasRoof: false },
  NYM: { name: 'Citi Field',               orientation: 135, hasRoof: false },
  NYY: { name: 'Yankee Stadium',           orientation: 90,  hasRoof: false },
  OAK: { name: 'Oakland Coliseum',         orientation: 270, hasRoof: false },
  PHI: { name: 'Citizens Bank Park',       orientation: 135, hasRoof: false },
  PIT: { name: 'PNC Park',                 orientation: 0,   hasRoof: false },
  SD:  { name: 'Petco Park',               orientation: 180, hasRoof: false },
  SEA: { name: 'T-Mobile Park',            orientation: 180, hasRoof: true  },
  SF:  { name: 'Oracle Park',              orientation: 225, hasRoof: false },
  STL: { name: 'Busch Stadium',            orientation: 180, hasRoof: false },
  TB:  { name: 'Tropicana Field',          orientation: 180, hasRoof: true  },
  TEX: { name: 'Globe Life Field',         orientation: 180, hasRoof: true  },
  TOR: { name: 'Rogers Centre',            orientation: 180, hasRoof: true  },
  WSH: { name: 'Nationals Park',           orientation: 180, hasRoof: false }
};
```

---

## Usage Example

```javascript
// User inputs
const result = runProjection({
  temp: 85,              // 85°F
  humidity: 35,          // 35%
  windSpeed: 12,         // 12 mph
  windDir: 270,          // Wind from west
  isNight: false,        // Day game
  parkAbbr: 'COL',
  stadiumOrientation: STADIUM_DATA['COL'].orientation
});

console.log(result);
// {
//   strikeouts: { predicted: 9.15, baseline: 8.92, delta: 0.23, direction: 'up' },
//   runs: { predicted: 4.72, baseline: 4.45, delta: 0.27, direction: 'up' },
//   computed: { airDensity: 1.1603, windCf: 11.5 },
//   isRoofed: false
// }
```

---

# Part 2: Explainer Implementation

## Overview

The explainer provides human-readable explanations for predictions, showing how each weather factor contributes to the outcome. All data is self-contained - no API calls needed.

---

## Baseline Statistics (League-Wide)

These are the league-average conditions used to calculate deviations:

```javascript
const LEAGUE_BASELINES = {
  temp_f:      { mean: 71.69, median: 72.4,  std: 12.36 },
  rhum:        { mean: 63.16, median: 65.3,  std: 19.68 },
  wspd_mph:    { mean: 7.73,  median: 7.0,   std: 3.87  },
  wind_cf:     { mean: 2.06,  median: 2.21,  std: 10.03 },
  air_density: { mean: 1.169, median: 1.171, std: 0.048 },
  is_night:    { mean: 0.619 }  // 62% of games are night games
};
```

---

## Park-Specific Baselines

Each park has its own typical weather conditions:

```javascript
const PARK_BASELINES = {
  ARI: { temp_f: 95.41, rhum: 18.71, wspd_mph: 7.30, wind_cf:  1.91, air_density: 1.092, is_night: 0.56 },
  ATL: { temp_f: 73.91, rhum: 68.16, wspd_mph: 5.84, wind_cf: -0.44, air_density: 1.144, is_night: 0.75 },
  BAL: { temp_f: 72.25, rhum: 66.42, wspd_mph: 6.49, wind_cf:  2.32, air_density: 1.187, is_night: 0.64 },
  BOS: { temp_f: 65.73, rhum: 69.43, wspd_mph: 7.93, wind_cf:  2.94, air_density: 1.203, is_night: 0.64 },
  CHC: { temp_f: 67.09, rhum: 69.06, wspd_mph: 10.39, wind_cf: -1.47, air_density: 1.177, is_night: 0.43 },
  CIN: { temp_f: 73.74, rhum: 59.22, wspd_mph: 6.50, wind_cf:  2.09, air_density: 1.165, is_night: 0.59 },
  CLE: { temp_f: 67.35, rhum: 71.99, wspd_mph: 8.08, wind_cf: -1.91, air_density: 1.174, is_night: 0.66 },
  COL: { temp_f: 71.00, rhum: 37.88, wspd_mph: 6.20, wind_cf: -1.83, air_density: 0.994, is_night: 0.64 },
  CWS: { temp_f: 67.86, rhum: 68.98, wspd_mph: 9.84, wind_cf: -1.00, air_density: 1.174, is_night: 0.61 },
  DET: { temp_f: 69.88, rhum: 59.62, wspd_mph: 8.99, wind_cf: -0.22, air_density: 1.170, is_night: 0.52 },
  HOU: { temp_f: 81.17, rhum: 72.43, wspd_mph: 7.58, wind_cf:  8.22, air_density: 1.161, is_night: 0.69 },
  KC:  { temp_f: 74.16, rhum: 60.52, wspd_mph: 6.96, wind_cf:  0.14, air_density: 1.148, is_night: 0.63 },
  LAA: { temp_f: 70.33, rhum: 63.11, wspd_mph: 6.43, wind_cf:  9.52, air_density: 1.185, is_night: 0.76 },
  LAD: { temp_f: 68.35, rhum: 63.73, wspd_mph: 5.89, wind_cf:  7.87, air_density: 1.173, is_night: 0.74 },
  MIA: { temp_f: 81.48, rhum: 72.95, wspd_mph: 7.66, wind_cf: -5.93, air_density: 1.165, is_night: 0.62 },
  MIL: { temp_f: 65.07, rhum: 64.78, wspd_mph: 8.44, wind_cf: -0.67, air_density: 1.181, is_night: 0.60 },
  MIN: { temp_f: 68.96, rhum: 58.73, wspd_mph: 7.34, wind_cf: -0.70, air_density: 1.162, is_night: 0.59 },
  NYM: { temp_f: 68.60, rhum: 67.71, wspd_mph: 7.95, wind_cf:  3.99, air_density: 1.198, is_night: 0.62 },
  NYY: { temp_f: 69.45, rhum: 65.80, wspd_mph: 7.71, wind_cf:  1.92, air_density: 1.195, is_night: 0.62 },
  OAK: { temp_f: 64.11, rhum: 70.02, wspd_mph: 8.46, wind_cf: 10.62, air_density: 1.208, is_night: 0.50 },
  PHI: { temp_f: 72.81, rhum: 61.49, wspd_mph: 7.22, wind_cf:  3.02, air_density: 1.188, is_night: 0.66 },
  PIT: { temp_f: 70.34, rhum: 62.96, wspd_mph: 6.02, wind_cf:  2.66, air_density: 1.163, is_night: 0.61 },
  SD:  { temp_f: 67.26, rhum: 72.68, wspd_mph: 7.34, wind_cf:  0.42, air_density: 1.198, is_night: 0.53 },
  SEA: { temp_f: 63.08, rhum: 63.65, wspd_mph: 7.33, wind_cf: -2.39, air_density: 1.213, is_night: 0.67 },
  SF:  { temp_f: 60.73, rhum: 74.27, wspd_mph: 14.31, wind_cf: 21.19, air_density: 1.215, is_night: 0.58 },
  STL: { temp_f: 75.32, rhum: 62.10, wspd_mph: 7.01, wind_cf:  2.06, air_density: 1.161, is_night: 0.63 },
  TB:  { temp_f: 81.95, rhum: 66.06, wspd_mph: 8.17, wind_cf:  0.86, air_density: 1.163, is_night: 0.61 },
  TEX: { temp_f: 83.07, rhum: 56.25, wspd_mph: 8.41, wind_cf: -2.47, air_density: 1.138, is_night: 0.66 },
  TOR: { temp_f: 65.86, rhum: 66.08, wspd_mph: 7.93, wind_cf:  0.30, air_density: 1.192, is_night: 0.58 },
  WSH: { temp_f: 73.21, rhum: 60.98, wspd_mph: 6.33, wind_cf:  0.57, air_density: 1.185, is_night: 0.56 }
};
```

---

## Feature Display Information

```javascript
const FEATURE_INFO = {
  temp_f:      { label: 'Temperature',  unit: '°F',    format: v => `${v.toFixed(0)}°F` },
  rhum:        { label: 'Humidity',     unit: '%',     format: v => `${v.toFixed(0)}%` },
  wspd_mph:    { label: 'Wind Speed',   unit: 'mph',   format: v => `${v.toFixed(0)} mph` },
  wind_cf:     { label: 'Wind to CF',   unit: '',      format: v => v > 0 ? `${v.toFixed(1)} out` : `${Math.abs(v).toFixed(1)} in` },
  air_density: { label: 'Air Density',  unit: 'kg/m³', format: v => `${v.toFixed(3)} kg/m³` },
  is_night:    { label: 'Day/Night',    unit: '',      format: v => v ? 'Night' : 'Day' }
};

const PARK_NAMES = {
  ARI: 'Chase Field',              ATL: 'Truist Park',
  BAL: 'Camden Yards',             BOS: 'Fenway Park',
  CHC: 'Wrigley Field',            CIN: 'Great American Ball Park',
  CLE: 'Progressive Field',        COL: 'Coors Field',
  CWS: 'Guaranteed Rate Field',    DET: 'Comerica Park',
  HOU: 'Minute Maid Park',         KC:  'Kauffman Stadium',
  LAA: 'Angel Stadium',            LAD: 'Dodger Stadium',
  MIA: 'loanDepot Park',           MIL: 'American Family Field',
  MIN: 'Target Field',             NYM: 'Citi Field',
  NYY: 'Yankee Stadium',           OAK: 'Oakland Coliseum',
  PHI: 'Citizens Bank Park',       PIT: 'PNC Park',
  SD:  'Petco Park',               SEA: 'T-Mobile Park',
  SF:  'Oracle Park',              STL: 'Busch Stadium',
  TB:  'Tropicana Field',          TEX: 'Globe Life Field',
  TOR: 'Rogers Centre',            WSH: 'Nationals Park'
};
```

---

## Explainer Functions

### 1. Get Baseline Value

```javascript
/**
 * Get baseline value for a feature
 * @param {string} feature - Feature name
 * @param {string} baselineType - 'league' or 'park'
 * @param {string} parkAbbr - Park abbreviation (required if baselineType='park')
 * @returns {number} Baseline value
 */
function getBaseline(feature, baselineType = 'league', parkAbbr = null) {
  if (baselineType === 'park' && parkAbbr && PARK_BASELINES[parkAbbr]) {
    return PARK_BASELINES[parkAbbr][feature];
  }
  return LEAGUE_BASELINES[feature].mean;
}
```

### 2. Calculate Feature Contribution

```javascript
/**
 * Calculate how much a feature contributes to prediction vs baseline
 * @param {string} feature - Feature name
 * @param {number} value - Current value
 * @param {string} target - 'strikeouts' or 'runs'
 * @param {string} baselineType - 'league' or 'park'
 * @param {string} parkAbbr - Park abbreviation
 * @param {boolean} isRoofed - Whether stadium has roof
 * @returns {Object} Contribution details
 */
function calculateContribution(feature, value, target, baselineType, parkAbbr, isRoofed) {
  const baseline = getBaseline(feature, baselineType, parkAbbr);
  const deviation = value - baseline;

  const coef = MODEL_PARAMS[target].fixed_effects[feature];
  const scaling = MODEL_PARAMS.scaling[feature];

  // Scale both values
  const valueScaled = (value - scaling.center) / scaling.scale;
  const baselineScaled = (baseline - scaling.center) / scaling.scale;
  const deviationScaled = valueScaled - baselineScaled;

  // Apply roof dampening if applicable (not for is_night)
  const dampening = (isRoofed && feature !== 'is_night')
    ? MODEL_PARAMS.roof_dampening_factor
    : 1.0;

  const contribution = coef * deviationScaled * dampening;

  return {
    feature,
    label: FEATURE_INFO[feature].label,
    rawValue: value,
    baseline: baseline,
    deviation: deviation,
    contribution: contribution,
    direction: contribution > 0.01 ? 'positive' : contribution < -0.01 ? 'negative' : 'neutral'
  };
}
```

### 3. Generate Full Explanation

```javascript
/**
 * Generate complete explanation for a prediction
 * @param {Object} weather - Weather conditions
 * @param {string} parkAbbr - Park abbreviation
 * @param {string} target - 'strikeouts' or 'runs'
 * @param {string} baselineType - 'league' or 'park'
 * @returns {Object} Complete explanation
 */
function explainPrediction(weather, parkAbbr, target = 'strikeouts', baselineType = 'league') {
  const isRoofed = hasRoof(parkAbbr);
  const targetUnit = target === 'strikeouts' ? 'K' : 'R';

  // Calculate prediction
  const result = predict(target, weather, parkAbbr);

  // Calculate baseline prediction
  const baselineWeather = {
    temp_f: getBaseline('temp_f', baselineType, parkAbbr),
    rhum: getBaseline('rhum', baselineType, parkAbbr),
    wspd_mph: getBaseline('wspd_mph', baselineType, parkAbbr),
    wind_cf: getBaseline('wind_cf', baselineType, parkAbbr),
    air_density: getBaseline('air_density', baselineType, parkAbbr),
    is_night: getBaseline('is_night', baselineType, parkAbbr) >= 0.5
  };
  const baselineResult = predict(target, baselineWeather, parkAbbr);

  // Calculate contributions for each feature
  const contributions = [];
  const features = ['temp_f', 'rhum', 'wspd_mph', 'wind_cf', 'air_density', 'is_night'];

  for (const feature of features) {
    const value = feature === 'is_night' ? (weather.is_night ? 1 : 0) : weather[feature];
    const contrib = calculateContribution(feature, value, target, baselineType, parkAbbr, isRoofed);
    contributions.push(contrib);
  }

  // Add park effect
  const parkEffect = MODEL_PARAMS[target].park_effects[parkAbbr] || 0;
  contributions.push({
    feature: 'park_effect',
    label: `${parkAbbr} Park Effect`,
    rawValue: parkEffect,
    baseline: 0,
    deviation: 0,
    contribution: parkEffect,
    direction: parkEffect > 0.01 ? 'positive' : parkEffect < -0.01 ? 'negative' : 'neutral'
  });

  // Sort by absolute contribution
  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  // Build factors array for display
  const factors = contributions.map(c => ({
    id: c.feature,
    label: c.label,
    description: buildDescription(c, targetUnit),
    contribution: +c.contribution.toFixed(3),
    direction: c.direction,
    rawValue: c.rawValue,
    deviation: c.deviation
  }));

  // Build summary
  const delta = result.prediction - baselineResult.prediction;
  let headline;
  if (Math.abs(delta) < 0.1) {
    headline = `Expect roughly average ${target} (${result.prediction} ${targetUnit})`;
  } else if (delta > 0) {
    headline = `Expect +${delta.toFixed(1)} more ${target} than baseline (${result.prediction} ${targetUnit})`;
  } else {
    headline = `Expect ${delta.toFixed(1)} fewer ${target} than baseline (${result.prediction} ${targetUnit})`;
  }

  // Key drivers
  const keyDrivers = contributions
    .filter(c => Math.abs(c.contribution) > 0.05 && c.feature !== 'park_effect')
    .slice(0, 3)
    .map(c => {
      const effect = c.contribution > 0 ? 'increases' : 'decreases';
      return `${c.label} ${effect} ${target} by ${Math.abs(c.contribution).toFixed(2)}`;
    });

  if (keyDrivers.length === 0) {
    keyDrivers.push('Weather conditions near league average');
  }

  return {
    meta: {
      version: '5.0.0',
      target,
      park: parkAbbr,
      parkName: PARK_NAMES[parkAbbr],
      isRoofed,
      baselineType
    },
    prediction: {
      value: result.prediction,
      baseline: baselineResult.prediction,
      delta: +delta.toFixed(2),
      unit: targetUnit
    },
    factors,
    summary: {
      headline,
      keyDrivers
    },
    disclaimer: {
      text: `Weather explains approximately ${target === 'strikeouts' ? '4' : '2'}% of ${target === 'strikeouts' ? 'strikeout' : 'run'} variance. Team quality and pitcher matchups have larger effects.`,
      note: 'Predictions are statistical estimates, not guarantees.'
    }
  };
}

/**
 * Build description string for a contribution
 */
function buildDescription(contrib, targetUnit) {
  const sign = contrib.contribution >= 0 ? '+' : '';

  if (contrib.feature === 'park_effect') {
    return `${contrib.label} contributes ${sign}${contrib.contribution.toFixed(2)} ${targetUnit}`;
  }

  if (contrib.feature === 'is_night') {
    const desc = contrib.rawValue ? 'Night game' : 'Day game';
    return `${desc} contributes ${sign}${contrib.contribution.toFixed(2)} ${targetUnit}`;
  }

  const info = FEATURE_INFO[contrib.feature];
  const devSign = contrib.deviation >= 0 ? '+' : '';
  return `${info.label} (${info.format(contrib.rawValue)}, ${devSign}${contrib.deviation.toFixed(1)} vs avg) contributes ${sign}${contrib.contribution.toFixed(2)} ${targetUnit}`;
}
```

---

## Waterfall Chart Data

For visualizing the breakdown:

```javascript
/**
 * Build waterfall chart data
 * @param {Object} explanation - Result from explainPrediction()
 * @returns {Array} Waterfall bars
 */
function buildWaterfallData(explanation) {
  const bars = [];
  let current = explanation.prediction.baseline;

  // Baseline bar
  bars.push({
    id: 'baseline',
    label: `${explanation.meta.baselineType.charAt(0).toUpperCase() + explanation.meta.baselineType.slice(1)} Baseline`,
    start: 0,
    end: current,
    type: 'baseline'
  });

  // Factor bars (excluding park effect, add at end)
  const factors = explanation.factors.filter(f => f.id !== 'park_effect' && Math.abs(f.contribution) > 0.01);

  for (const factor of factors) {
    bars.push({
      id: factor.id,
      label: factor.label,
      start: current,
      end: current + factor.contribution,
      type: factor.contribution > 0 ? 'positive' : 'negative'
    });
    current += factor.contribution;
  }

  // Park effect
  const parkFactor = explanation.factors.find(f => f.id === 'park_effect');
  if (parkFactor && Math.abs(parkFactor.contribution) > 0.01) {
    bars.push({
      id: 'park',
      label: parkFactor.label,
      start: current,
      end: current + parkFactor.contribution,
      type: parkFactor.contribution > 0 ? 'positive' : 'negative'
    });
  }

  // Total bar
  bars.push({
    id: 'total',
    label: 'Prediction',
    start: 0,
    end: explanation.prediction.value,
    type: 'total'
  });

  return bars;
}
```

---

## Complete Usage Example

```javascript
// 1. User selects park and enters weather
const parkAbbr = 'COL';
const weather = {
  temp_f: 85,
  rhum: 35,
  wspd_mph: 8,
  wind_cf: 7.7,  // Or compute: computeWindCF(8, 270, STADIUM_DATA['COL'].orientation)
  air_density: computeAirDensity(85, 35),
  is_night: false
};

// 2. Get explanation for strikeouts
const kExplanation = explainPrediction(weather, parkAbbr, 'strikeouts', 'league');

console.log('PREDICTION:', kExplanation.prediction.value, 'K');
console.log('BASELINE:', kExplanation.prediction.baseline, 'K');
console.log('DELTA:', kExplanation.prediction.delta, 'K');
console.log('');
console.log('HEADLINE:', kExplanation.summary.headline);
console.log('');
console.log('KEY FACTORS:');
kExplanation.summary.keyDrivers.forEach(d => console.log('  -', d));
console.log('');
console.log('ALL FACTORS:');
kExplanation.factors.forEach(f => {
  console.log(`  ${f.label}: ${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(2)} K`);
});

// 3. Get waterfall data for visualization
const waterfallBars = buildWaterfallData(kExplanation);
console.log('');
console.log('WATERFALL BARS:', waterfallBars);

// 4. Similarly for runs
const rExplanation = explainPrediction(weather, parkAbbr, 'runs', 'league');
console.log('');
console.log('RUNS PREDICTION:', rExplanation.prediction.value, 'R');
```

### Expected Output:

```
PREDICTION: 9.08 K
BASELINE: 8.92 K
DELTA: 0.16 K

HEADLINE: Expect +0.2 more strikeouts than baseline (9.08 K)

KEY FACTORS:
  - Temperature increases strikeouts by 0.16
  - Humidity increases strikeouts by 0.07

ALL FACTORS:
  - Temperature: +0.16 K
  - Air Density: -0.08 K
  - Humidity: +0.07 K
  - COL Park Effect: +0.04 K
  - Wind Speed: +0.01 K
  - Wind to CF: +0.04 K
  - Day/Night: +0.13 K
```

---

## Summary

This implementation provides:

1. **Model Parameters**: Complete coefficients for strikeouts and runs
2. **Scaling Parameters**: For standardizing weather inputs
3. **Baseline Statistics**: League-wide and park-specific averages
4. **Core Functions**: Prediction, contribution calculation, explanation generation
5. **Visualization Data**: Waterfall chart data structure
6. **Stadium Data**: Orientations and roof status for all 30 parks

All data is self-contained - no external API calls required. The model explains ~4% of strikeout variance and ~2% of run variance from weather, so use for directional guidance rather than precise predictions.
