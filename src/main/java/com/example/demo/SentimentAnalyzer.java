package com.example.demo;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SentimentAnalyzer {
    private static final ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    public String analyze(String text) {
        try {
            // Python 스크립트 경로
            String scriptPath = getClass().getClassLoader().getResource("sentiment_analysis.py").getPath();
            scriptPath = scriptPath.replaceFirst("^/(.:/)", "$1"); // 경로 수정
            System.out.println("Script Path: " + scriptPath);

            // 명시적으로 Python 실행 파일 경로를 지정
            String pythonExecutable = "C:\\Users\\leehosang\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"; // 설치된 Python 경로로 변경
            ProcessBuilder processBuilder = new ProcessBuilder(pythonExecutable, scriptPath, text);
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();

            // Python 스크립트의 출력 읽기
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder result = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                result.append(line);
            }
            reader.close();

            // 오류 메시지 읽기
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            StringBuilder errorResult = new StringBuilder();
            while ((line = errorReader.readLine()) != null) {
                errorResult.append(line);
            }
            errorReader.close();

            // 프로세스 종료 코드 확인
            int exitCode = process.waitFor();
            if (exitCode == 0) {
                return result.toString();
            } else {
                System.err.println("Error occurred: " + errorResult.toString());
                return "Error occurred";
            }
        } catch (Exception e) {
            e.printStackTrace();
            return "Error occurred";
        }
    }

    public static void main(String[] args) {
        SentimentAnalyzer analyzer = new SentimentAnalyzer();
        String sentiment = analyzer.analyze("I love this product!");
        System.out.println("Sentiment: " + sentiment);
    }
}
