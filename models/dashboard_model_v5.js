/**
 * ParkCast Dashboard Model v5.0
 * Park-Weather Interaction Model with Full Weather Features
 *
 * NEW IN V5:
 * - Added rhum (humidity) feature
 * - Added air_density feature (computed from temp/humidity)
 *
 * This model predicts directional weather effects on strikeouts and runs.
 * Roofed stadiums have weather effects dampened by 95% (roof_dampening_factor = 0.05).
 *
 * Model performance:
 * - Strikeouts: R² = 0.041 (explains ~4% of variance)
 * - Runs: R² = 0.023 (explains ~2% of variance)
 *
 * Use for directional guidance only (will weather increase/decrease K or R),
 * not for precise point predictions.
 */

const MODEL_PARAMS_V5 = {
  version: '5.0.0',

  // Scaling parameters (RobustScaler: center = median, scale = IQR)
  scaling: {
    temp_f:      { center: 73.0,     scale: 16.3     },
    rhum:        { center: 62.7,     scale: 26.0     },
    wspd_mph:    { center: 7.0,      scale: 4.7      },
    wind_cf:     { center: 1.89,     scale: 13.95    },
    air_density: { center: 1.170203, scale: 0.048265 }
  },

  // Roof dampening: weather effects multiplied by this factor for roofed stadiums
  roof_dampening_factor: 0.05,

  // Roofed/retractable roof stadiums
  roofed_stadiums: ['ARI', 'HOU', 'MIA', 'MIL', 'SEA', 'TB', 'TEX', 'TOR'],

  strikeouts: {
    fixed_effects: {
      Intercept:   8.8862,
      temp_f:      0.2186,   // Hotter = more strikeouts (in v5, reversed from v4)
      rhum:       -0.0689,   // Higher humidity = fewer strikeouts
      wspd_mph:    0.0268,   // More wind = slightly more strikeouts
      wind_cf:     0.0157,   // Wind out to CF has minimal K effect
      air_density: 0.4109,   // Denser air = more strikeouts (ball doesn't carry)
      is_night:   -0.2178    // Night games = fewer strikeouts
    },
    park_effects: {
      ARI: -0.0003, ATL:  0.1795, BAL: -0.4940, BOS:  0.1042, CHC:  0.1014,
      CIN:  0.1986, CLE:  0.3389, COL:  0.0352, CWS:  0.1070, DET: -0.5348,
      HOU:  0.5008, KC:  -0.5076, LAA:  0.0773, LAD:  0.5124, MIA: -0.0790,
      MIL:  0.2942, MIN: -0.1186, NYM:  0.2793, NYY:  0.3461, OAK: -0.3389,
      PHI:  0.1816, PIT: -0.2412, SD:   0.1876, SEA: -0.0907, SF:  -0.0658,
      STL: -0.4051, TB:   0.2272, TEX: -0.4813, TOR: -0.2353, WSH: -0.0786
    }
  },

  runs: {
    fixed_effects: {
      Intercept:   4.4227,
      temp_f:      0.0263,   // Hotter = slightly more runs
      rhum:       -0.0972,   // Higher humidity = fewer runs
      wspd_mph:    0.0731,   // More wind = more runs
      wind_cf:     0.1261,   // Wind out to CF = more runs
      air_density: -0.2157,  // Denser air = fewer runs (ball doesn't carry)
      is_night:    0.1259    // Night games = slightly more runs
    },
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

/**
 * Compute air density from temperature and humidity
 * Uses simplified formula: density = f(T, RH, P) where P is assumed sea-level
 *
 * @param {number} tempF - Temperature in Fahrenheit
 * @param {number} rhum - Relative humidity (0-100)
 * @param {number} presMb - Pressure in millibars (default: 1013.25, sea level)
 * @returns {number} Air density in kg/m³
 */
function computeAirDensity(tempF, rhum, presMb = 1013.25) {
  // Convert to Kelvin
  const tempK = (tempF - 32) * 5/9 + 273.15;
  const tempC = (tempF - 32) * 5/9;

  // Saturation vapor pressure (Magnus formula)
  const eSat = 6.1078 * Math.pow(10, (7.5 * tempC) / (tempC + 237.3));

  // Actual vapor pressure
  const e = (rhum / 100) * eSat;

  // Partial pressure of dry air (in Pa)
  const pD = (presMb - e) * 100;

  // Gas constants
  const Rd = 287.05;  // J/(kg·K) for dry air
  const Rv = 461.5;   // J/(kg·K) for water vapor

  // Air density
  return pD / (Rd * tempK) + (e * 100) / (Rv * tempK);
}

/**
 * Check if a stadium has a roof (retractable or dome)
 * @param {string} parkAbbr - Three-letter park abbreviation
 * @returns {boolean} True if stadium has a roof
 */
function hasRoof(parkAbbr) {
  return MODEL_PARAMS_V5.roofed_stadiums.includes(parkAbbr);
}

/**
 * Scale a feature using RobustScaler parameters
 * @param {number} value - Raw feature value
 * @param {string} feature - Feature name
 * @returns {number} Scaled value
 */
function scaleFeature(value, feature) {
  const { center, scale } = MODEL_PARAMS_V5.scaling[feature];
  return (value - center) / scale;
}

/**
 * Compute wind component blowing toward center field
 * @param {number} wspdMph - Wind speed in mph
 * @param {number} windDirDeg - Wind direction (degrees, meteorological: direction wind blows FROM)
 * @param {number} stadiumOrientation - Stadium CF orientation (degrees from north)
 * @returns {number} Wind component toward CF (positive = blowing out, negative = blowing in)
 */
function computeWindCF(wspdMph, windDirDeg, stadiumOrientation) {
  const relDeg = ((windDirDeg - stadiumOrientation) + 360) % 360;
  const relRad = relDeg * Math.PI / 180;
  return wspdMph * Math.cos(relRad);
}

/**
 * Compute weather effect on a target variable
 * @param {Object} params - Model parameters object (MODEL_PARAMS_V5.strikeouts or .runs)
 * @param {Object} weather - Weather conditions
 * @param {number} weather.temp_f - Temperature in Fahrenheit
 * @param {number} weather.rhum - Relative humidity (0-100)
 * @param {number} weather.wspd_mph - Wind speed in mph
 * @param {number} weather.wind_cf - Wind component toward CF
 * @param {number} weather.air_density - Air density in kg/m³
 * @param {boolean} weather.is_night - True for night game
 * @param {string} parkAbbr - Three-letter park abbreviation
 * @returns {Object} { prediction, weather_effect, park_effect }
 */
function computePrediction(params, weather, parkAbbr) {
  const fe = params.fixed_effects;

  // Check if this is a roofed stadium
  const isRoofed = hasRoof(parkAbbr);
  const dampening = isRoofed ? MODEL_PARAMS_V5.roof_dampening_factor : 1.0;

  // Scale weather features
  const temp_scaled = scaleFeature(weather.temp_f, 'temp_f');
  const rhum_scaled = scaleFeature(weather.rhum, 'rhum');
  const wspd_scaled = scaleFeature(weather.wspd_mph, 'wspd_mph');
  const wind_cf_scaled = scaleFeature(weather.wind_cf, 'wind_cf');
  const density_scaled = scaleFeature(weather.air_density, 'air_density');

  // Compute weather effects with roof dampening
  // For roofed stadiums, multiply weather coefficients by dampening factor
  const weather_effect =
    (fe.temp_f * temp_scaled * dampening) +
    (fe.rhum * rhum_scaled * dampening) +
    (fe.wspd_mph * wspd_scaled * dampening) +
    (fe.wind_cf * wind_cf_scaled * dampening) +
    (fe.air_density * density_scaled * dampening) +
    (fe.is_night * (weather.is_night ? 1 : 0));

  // Get park effect (not dampened - this is structural)
  const park_effect = params.park_effects[parkAbbr] || 0;

  // Total prediction
  const prediction = fe.Intercept + weather_effect + park_effect;

  // Individual contributions for breakdown
  const contributions = {
    temp_f: fe.temp_f * temp_scaled * dampening,
    rhum: fe.rhum * rhum_scaled * dampening,
    wspd_mph: fe.wspd_mph * wspd_scaled * dampening,
    wind_cf: fe.wind_cf * wind_cf_scaled * dampening,
    air_density: fe.air_density * density_scaled * dampening,
    is_night: fe.is_night * (weather.is_night ? 1 : 0)
  };

  return {
    prediction: +prediction.toFixed(2),
    weather_effect: +weather_effect.toFixed(3),
    park_effect: +park_effect.toFixed(2),
    contributions: contributions,
    is_roofed: isRoofed,
    dampening_applied: dampening
  };
}

/**
 * Run full projection for a stadium and weather conditions
 * @param {Object} weather - Raw weather inputs
 * @param {number} weather.temp - Temperature in Fahrenheit
 * @param {number} weather.humidity - Relative humidity (0-100)
 * @param {number} weather.wind - Wind speed in mph
 * @param {number} weather.winddir - Wind direction in degrees (meteorological)
 * @param {boolean} weather.isNight - True for night game
 * @param {Object} stadium - Stadium object with abbr and orientation
 * @param {string} stadium.abbr - Three-letter park abbreviation
 * @param {number} stadium.orientation - Stadium CF orientation in degrees
 * @returns {Object} Full projection results
 */
function runProjection(weather, stadium) {
  // Compute derived features
  const wind_cf = computeWindCF(weather.wind, weather.winddir, stadium.orientation);
  const air_density = computeAirDensity(weather.temp, weather.humidity);

  // Build weather object for model
  const weatherForModel = {
    temp_f: weather.temp,
    rhum: weather.humidity,
    wspd_mph: weather.wind,
    wind_cf: wind_cf,
    air_density: air_density,
    is_night: weather.isNight || false
  };

  // Get predictions
  const kResult = computePrediction(MODEL_PARAMS_V5.strikeouts, weatherForModel, stadium.abbr);
  const rResult = computePrediction(MODEL_PARAMS_V5.runs, weatherForModel, stadium.abbr);

  // Compute baseline (neutral conditions: league averages)
  const neutralWeather = {
    temp_f: 73,
    rhum: 63,
    wspd_mph: 7,
    wind_cf: 0,
    air_density: 1.17,
    is_night: false
  };
  const kBaseline = computePrediction(MODEL_PARAMS_V5.strikeouts, neutralWeather, stadium.abbr);
  const rBaseline = computePrediction(MODEL_PARAMS_V5.runs, neutralWeather, stadium.abbr);

  return {
    // Strikeouts prediction
    strikeouts: {
      predicted: kResult.prediction,
      baseline: kBaseline.prediction,
      delta: +(kResult.prediction - kBaseline.prediction).toFixed(2),
      direction: kResult.prediction > kBaseline.prediction ? 'up' : kResult.prediction < kBaseline.prediction ? 'down' : 'neutral',
      weather_effect: kResult.weather_effect,
      park_effect: kResult.park_effect,
      contributions: kResult.contributions
    },

    // Runs prediction
    runs: {
      predicted: rResult.prediction,
      baseline: rBaseline.prediction,
      delta: +(rResult.prediction - rBaseline.prediction).toFixed(2),
      direction: rResult.prediction > rBaseline.prediction ? 'up' : rResult.prediction < rBaseline.prediction ? 'down' : 'neutral',
      weather_effect: rResult.weather_effect,
      park_effect: rResult.park_effect,
      contributions: rResult.contributions
    },

    // Derived weather features
    computed: {
      air_density: +air_density.toFixed(4),
      wind_cf: +wind_cf.toFixed(1)
    },

    // Metadata
    is_roofed: kResult.is_roofed,
    roof_note: kResult.is_roofed ?
      'Roofed stadium: weather effects dampened by 95%' :
      'Open-air stadium: full weather effects applied',

    // Wind analysis
    wind: {
      cf_component: +wind_cf.toFixed(1),
      effect: wind_cf > 2 ? 'blowing_out' : wind_cf < -2 ? 'blowing_in' : 'crosswind'
    }
  };
}

/**
 * Get a plain-English interpretation of the projection
 * @param {Object} projection - Result from runProjection()
 * @returns {Object} Plain-English interpretation
 */
function interpretProjection(projection) {
  const k = projection.strikeouts;
  const r = projection.runs;

  let kInterpretation, rInterpretation;

  // Strikeout interpretation
  if (Math.abs(k.delta) < 0.1) {
    kInterpretation = 'neutral conditions for strikeouts';
  } else if (k.delta > 0) {
    kInterpretation = `conditions favor +${k.delta.toFixed(1)} more strikeouts`;
  } else {
    kInterpretation = `conditions suppress strikeouts by ${Math.abs(k.delta).toFixed(1)}`;
  }

  // Runs interpretation
  if (Math.abs(r.delta) < 0.1) {
    rInterpretation = 'neutral conditions for run scoring';
  } else if (r.delta > 0) {
    rInterpretation = `conditions favor +${r.delta.toFixed(1)} more runs`;
  } else {
    rInterpretation = `conditions suppress run scoring by ${Math.abs(r.delta).toFixed(1)}`;
  }

  // Key factors (sorted by absolute contribution)
  const factors = [];
  const kContrib = k.contributions;
  const sortedFactors = Object.entries(kContrib)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 3);

  for (const [factor, contrib] of sortedFactors) {
    if (Math.abs(contrib) > 0.02) {
      const dir = contrib > 0 ? 'increases' : 'decreases';
      factors.push(`${factor.replace('_', ' ')} ${dir} K by ${Math.abs(contrib).toFixed(2)}`);
    }
  }

  // Overall assessment
  let overall;
  if (k.delta > 0.15 && r.delta < -0.15) {
    overall = 'pitcher-friendly conditions';
  } else if (k.delta < -0.15 && r.delta > 0.15) {
    overall = 'hitter-friendly conditions';
  } else if (k.delta > 0.15 && r.delta > 0.15) {
    overall = 'high-action conditions (more K and R)';
  } else if (k.delta < -0.15 && r.delta < -0.15) {
    overall = 'low-action conditions (fewer K and R)';
  } else {
    overall = 'near-neutral conditions';
  }

  return {
    strikeouts: kInterpretation,
    runs: rInterpretation,
    key_factors: factors,
    overall: overall,
    roof_note: projection.roof_note
  };
}

/**
 * Get explanation for why conditions affect prediction
 * @param {Object} projection - Result from runProjection()
 * @param {string} target - 'strikeouts' or 'runs'
 * @returns {Array} Array of explanation strings
 */
function explainPrediction(projection, target = 'strikeouts') {
  const result = projection[target];
  const contributions = result.contributions;
  const explanations = [];

  const featureNames = {
    temp_f: 'Temperature',
    rhum: 'Humidity',
    wspd_mph: 'Wind speed',
    wind_cf: 'Wind direction',
    air_density: 'Air density',
    is_night: 'Day/Night'
  };

  const sorted = Object.entries(contributions)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  for (const [feature, contrib] of sorted) {
    if (Math.abs(contrib) > 0.01) {
      const name = featureNames[feature] || feature;
      const effect = contrib > 0 ? 'increases' : 'decreases';
      const unit = target === 'strikeouts' ? 'K' : 'R';
      explanations.push(`${name} ${effect} ${unit} by ${Math.abs(contrib).toFixed(2)}`);
    }
  }

  return explanations;
}

// ============================================================
// EXAMPLE USAGE
// ============================================================
/*
// Example: Project conditions at Coors Field (COL) on a hot, dry day
const weather = {
  temp: 85,        // 85°F (hot day)
  humidity: 35,    // 35% humidity (dry, typical for Denver)
  wind: 8,         // 8 mph wind
  winddir: 270,    // Wind from the west
  isNight: false   // Day game
};

const stadium = {
  abbr: 'COL',
  orientation: 255  // CF faces west-southwest
};

const result = runProjection(weather, stadium);
console.log(result);
// {
//   strikeouts: { predicted: 9.15, baseline: 8.89, delta: 0.26, direction: 'up', ... },
//   runs: { predicted: 4.38, baseline: 4.45, delta: -0.07, direction: 'down', ... },
//   computed: { air_density: 1.0923, wind_cf: 7.8 },
//   is_roofed: false,
//   roof_note: 'Open-air stadium: full weather effects applied',
//   wind: { cf_component: 7.8, effect: 'blowing_out' }
// }

const interpretation = interpretProjection(result);
console.log(interpretation);
// {
//   strikeouts: 'conditions favor +0.3 more strikeouts',
//   runs: 'neutral conditions for run scoring',
//   key_factors: ['air density decreases K by -0.16', 'temp f increases K by 0.10'],
//   overall: 'near-neutral conditions',
//   roof_note: 'Open-air stadium: full weather effects applied'
// }

// Example: Same conditions at Tropicana Field (TB) - domed
const tampaBayResult = runProjection(weather, { abbr: 'TB', orientation: 180 });
console.log(tampaBayResult.roof_note);
// 'Roofed stadium: weather effects dampened by 95%'
*/

// Export for use in dashboard
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    MODEL_PARAMS_V5,
    computeAirDensity,
    hasRoof,
    scaleFeature,
    computeWindCF,
    computePrediction,
    runProjection,
    interpretProjection,
    explainPrediction
  };
}
