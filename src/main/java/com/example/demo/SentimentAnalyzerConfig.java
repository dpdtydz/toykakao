package com.example.demo;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SentimentAnalyzerConfig {
    @Bean
    public SentimentAnalyzer sentimentAnalyzer() {
        return new SentimentAnalyzer();
    }
}
