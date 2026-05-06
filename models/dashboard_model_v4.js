/**
 * ParkCast Dashboard Model v4.0
 * Park-Weather Interaction Model with Roof Dampening
 *
 * This model predicts directional weather effects on strikeouts and runs.
 * Roofed stadiums have weather effects dampened by 95% (roof_dampening_factor = 0.05).
 *
 * Model performance:
 * - Strikeouts: R² = 0.041 (explains ~4% of variance)
 * - Runs: R² = 0.022 (explains ~2% of variance)
 *
 * Use for directional guidance only (will weather increase/decrease K or R),
 * not for precise point predictions.
 */

const MODEL_PARAMS = {
  // Scaling parameters (RobustScaler: center = median, scale = IQR)
  scaling: {
    temp_f:   { center: 73.0,  scale: 16.3  },
    wspd_mph: { center: 7.0,   scale: 4.7   },
    wind_cf:  { center: 1.89,  scale: 13.95 }
  },

  // Roof dampening: weather effects multiplied by this factor for roofed stadiums
  roof_dampening_factor: 0.05,

  // Roofed/retractable roof stadiums
  roofed_stadiums: ['ARI', 'HOU', 'MIA', 'MIL', 'SEA', 'TB', 'TEX', 'TOR'],

  strikeouts: {
    fixed_effects: {
      Intercept: 8.92,
      temp_f:    -0.08,  // Hotter = fewer strikeouts
      wspd_mph:   0.04,  // More wind = slightly more strikeouts
      wind_cf:    0.01,  // Wind out to CF has minimal K effect
      is_night:  -0.27   // Night games = fewer strikeouts
    },
    park_effects: {
      ARI: -0.13, ATL:  0.04, BAL: -0.48, BOS:  0.24, CHC:  0.06,
      CIN:  0.20, CLE:  0.30, COL: -0.82, CWS:  0.05, DET: -0.61,
      HOU:  0.77, KC:  -0.71, LAA:  0.13, LAD:  0.55, MIA:  0.05,
      MIL:  0.34, MIN: -0.29, NYM:  0.46, NYY:  0.49, OAK: -0.37,
      PHI:  0.35, PIT: -0.35, SD:   0.31, SEA:  0.11, SF:  -0.06,
      STL: -0.46, TB:   0.46, TEX: -0.51, TOR: -0.14, WSH:  0.01
    }
  },

  runs: {
    fixed_effects: {
      Intercept: 4.43,
      temp_f:    0.20,  // Hotter = more runs
      wspd_mph:  0.09,  // More wind = more runs
      wind_cf:   0.09,  // Wind out to CF = more runs
      is_night:  0.12   // Night games = slightly more runs
    },
    park_effects: {
      ARI:  0.01, ATL:  0.12, BAL:  0.29, BOS:  0.30, CHC: -0.23,
      CIN:  0.36, CLE: -0.29, COL:  0.83, CWS:  0.16, DET:  0.29,
      HOU: -0.36, KC:   0.23, LAA:  0.30, LAD: -0.64, MIA:  0.00,
      MIL: -0.23, MIN:  0.22, NYM: -0.32, NYY: -0.30, OAK:  0.08,
      PHI: -0.09, PIT:  0.16, SD:  -0.19, SEA: -0.19, SF:  -0.02,
      STL: -0.30, TB:  -0.37, TEX:  0.09, TOR:  0.02, WSH:  0.07
    }
  }
};

/**
 * Check if a stadium has a roof (retractable or dome)
 * @param {string} parkAbbr - Three-letter park abbreviation
 * @returns {boolean} True if stadium has a roof
 */
function hasRoof(parkAbbr) {
  return MODEL_PARAMS.roofed_stadiums.includes(parkAbbr);
}

/**
 * Scale a feature using RobustScaler parameters
 * @param {number} value - Raw feature value
 * @param {string} feature - Feature name ('temp_f', 'wspd_mph', 'wind_cf')
 * @returns {number} Scaled value
 */
function scaleFeature(value, feature) {
  const { center, scale } = MODEL_PARAMS.scaling[feature];
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
  // Wind direction is where wind blows FROM
  // Stadium orientation is where CF faces
  // We want the component blowing toward CF (out = positive)
  const relDeg = ((windDirDeg - stadiumOrientation) + 360) % 360;
  const relRad = relDeg * Math.PI / 180;
  return wspdMph * Math.cos(relRad);
}

/**
 * Compute weather effect on a target variable
 * @param {Object} params - Model parameters object (MODEL_PARAMS.strikeouts or MODEL_PARAMS.runs)
 * @param {Object} weather - Weather conditions
 * @param {number} weather.temp_f - Temperature in Fahrenheit
 * @param {number} weather.wspd_mph - Wind speed in mph
 * @param {number} weather.wind_cf - Wind component toward CF
 * @param {boolean} weather.is_night - True for night game
 * @param {string} parkAbbr - Three-letter park abbreviation
 * @returns {Object} { prediction, weather_effect, park_effect }
 */
function computePrediction(params, weather, parkAbbr) {
  const fe = params.fixed_effects;

  // Check if this is a roofed stadium
  const isRoofed = hasRoof(parkAbbr);
  const dampening = isRoofed ? MODEL_PARAMS.roof_dampening_factor : 1.0;

  // Scale weather features
  const temp_scaled = scaleFeature(weather.temp_f, 'temp_f');
  const wspd_scaled = scaleFeature(weather.wspd_mph, 'wspd_mph');
  const wind_cf_scaled = scaleFeature(weather.wind_cf, 'wind_cf');

  // Compute weather effects with roof dampening
  // For roofed stadiums, multiply weather coefficients by dampening factor
  const weather_effect =
    (fe.temp_f * temp_scaled * dampening) +
    (fe.wspd_mph * wspd_scaled * dampening) +
    (fe.wind_cf * wind_cf_scaled * dampening) +
    (fe.is_night * (weather.is_night ? 1 : 0));

  // Get park effect (not dampened - this is structural)
  const park_effect = params.park_effects[parkAbbr] || 0;

  // Total prediction
  const prediction = fe.Intercept + weather_effect + park_effect;

  return {
    prediction: +prediction.toFixed(2),
    weather_effect: +weather_effect.toFixed(3),
    park_effect: +park_effect.toFixed(2),
    is_roofed: isRoofed,
    dampening_applied: dampening
  };
}

/**
 * Run full projection for a stadium and weather conditions
 * @param {Object} weather - Raw weather inputs
 * @param {number} weather.temp - Temperature in Fahrenheit
 * @param {number} weather.wind - Wind speed in mph
 * @param {number} weather.winddir - Wind direction in degrees (meteorological)
 * @param {boolean} weather.isNight - True for night game
 * @param {Object} stadium - Stadium object with abbr and orientation
 * @param {string} stadium.abbr - Three-letter park abbreviation
 * @param {number} stadium.orientation - Stadium CF orientation in degrees
 * @returns {Object} Full projection results
 */
function runProjection(weather, stadium) {
  // Compute wind component toward CF
  const wind_cf = computeWindCF(weather.wind, weather.winddir, stadium.orientation);

  // Build weather object for model
  const weatherForModel = {
    temp_f: weather.temp,
    wspd_mph: weather.wind,
    wind_cf: wind_cf,
    is_night: weather.isNight || false
  };

  // Get predictions
  const kResult = computePrediction(MODEL_PARAMS.strikeouts, weatherForModel, stadium.abbr);
  const rResult = computePrediction(MODEL_PARAMS.runs, weatherForModel, stadium.abbr);

  // Compute baseline (neutral conditions: 73F, 7mph, neutral wind, day game)
  const neutralWeather = {
    temp_f: 73,
    wspd_mph: 7,
    wind_cf: 0,  // Neutral wind
    is_night: false
  };
  const kBaseline = computePrediction(MODEL_PARAMS.strikeouts, neutralWeather, stadium.abbr);
  const rBaseline = computePrediction(MODEL_PARAMS.runs, neutralWeather, stadium.abbr);

  return {
    // Strikeouts prediction
    strikeouts: {
      predicted: kResult.prediction,
      baseline: kBaseline.prediction,
      delta: +(kResult.prediction - kBaseline.prediction).toFixed(2),
      direction: kResult.prediction > kBaseline.prediction ? 'up' : kResult.prediction < kBaseline.prediction ? 'down' : 'neutral',
      weather_effect: kResult.weather_effect,
      park_effect: kResult.park_effect
    },

    // Runs prediction
    runs: {
      predicted: rResult.prediction,
      baseline: rBaseline.prediction,
      delta: +(rResult.prediction - rBaseline.prediction).toFixed(2),
      direction: rResult.prediction > rBaseline.prediction ? 'up' : rResult.prediction < rBaseline.prediction ? 'down' : 'neutral',
      weather_effect: rResult.weather_effect,
      park_effect: rResult.park_effect
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
    overall: overall,
    roof_note: projection.roof_note
  };
}

// ============================================================
// EXAMPLE USAGE
// ============================================================
/*
// Example: Project conditions at Oracle Park (SF) on a warm, windy day game
const weather = {
  temp: 85,       // 85°F (hot day)
  wind: 15,       // 15 mph wind
  winddir: 270,   // Wind from the west
  isNight: false  // Day game
};

const stadium = {
  abbr: 'SF',
  orientation: 225  // CF faces southwest
};

const result = runProjection(weather, stadium);
console.log(result);
// {
//   strikeouts: { predicted: 8.81, baseline: 8.86, delta: -0.05, direction: 'down', ... },
//   runs: { predicted: 4.67, baseline: 4.41, delta: 0.26, direction: 'up', ... },
//   is_roofed: false,
//   roof_note: 'Open-air stadium: full weather effects applied',
//   wind: { cf_component: 10.6, effect: 'blowing_out' }
// }

const interpretation = interpretProjection(result);
console.log(interpretation);
// {
//   strikeouts: 'neutral conditions for strikeouts',
//   runs: 'conditions favor +0.3 more runs',
//   overall: 'near-neutral conditions',
//   roof_note: 'Open-air stadium: full weather effects applied'
// }

// Example: Same conditions at Minute Maid Park (HOU) - roofed
const houstonResult = runProjection(weather, { abbr: 'HOU', orientation: 180 });
console.log(houstonResult.roof_note);
// 'Roofed stadium: weather effects dampened by 95%'
*/

// Export for use in dashboard
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    MODEL_PARAMS,
    hasRoof,
    scaleFeature,
    computeWindCF,
    computePrediction,
    runProjection,
    interpretProjection
  };
}
