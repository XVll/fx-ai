class FeatureExtractor:
    def extract_features(self, market_state):
        # Get buffer data
        buffer_1s = market_state['buffer_1s']
        rolling_1m = market_state['rolling_1m']
        rolling_5m = market_state['rolling_5m']

        # 1. Create HF features (60 time steps × 20 features)
        hf_features = np.zeros((60, 20))
        for i in range(min(60, len(buffer_1s))):
            second_data = buffer_1s[-(i + 1)]  # Most recent first
            bar = second_data['bar']

            # Fill HF features with normalized OHLCV, trade stats, etc.
            hf_features[i, 0] = bar['open']
            hf_features[i, 1] = bar['high']
            # ... and so on for all 20 features

        # 2. Create MF features (30 time steps × 15 features)
        mf_features = np.zeros((30, 15))
        for i in range(min(30, len(rolling_1m))):
            minute_data = rolling_1m[-(i + 1)]
            # Fill with 1-minute features...

        # 3. Create LF features (30 time steps × 10 features)
        lf_features = np.zeros((30, 10))
        for i in range(min(30, len(rolling_5m))):
            five_min_data = rolling_5m[-(i + 1)]
            # Fill with 5-minute features...

        # 4. Create static features (15 features)
        static_features = np.zeros(15)
        # Fill with current position, S/R levels, etc.

        return {
            'hf_features': hf_features,
            'mf_features': mf_features,
            'lf_features': lf_features,
            'static_features': static_features
        }