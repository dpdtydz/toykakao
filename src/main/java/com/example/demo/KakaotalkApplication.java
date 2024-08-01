package com.example.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@SpringBootApplication
@Controller
public class KakaotalkApplication {

    private final SentimentAnalyzer sentimentAnalyzer;
    private final ProgressWebSocketHandler progressWebSocketHandler;

    public KakaotalkApplication(SentimentAnalyzer sentimentAnalyzer, ProgressWebSocketHandler progressWebSocketHandler) {
        this.sentimentAnalyzer = sentimentAnalyzer;
        this.progressWebSocketHandler = progressWebSocketHandler;
    }

    public static void main(String[] args) {
        SpringApplication.run(KakaotalkApplication.class, args);
    }

    @GetMapping("/")
    public String home() {
        return "upload";
    }

    @PostMapping("/upload")
    public String uploadFile(@RequestParam("file") MultipartFile file, Model model) throws IOException, InterruptedException, ExecutionException {
        Map<String, Integer> userMessageCount = new ConcurrentHashMap<>();
        Map<String, Map<String, Integer>> userSentimentCounts = new ConcurrentHashMap<>();
        List<String> dates = new CopyOnWriteArrayList<>();
        boolean[] dateFound = {false};

        // 파일을 한 번에 읽어 메모리에 저장
        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(file.getInputStream(), "UTF-8"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        }

        int lineCount = lines.size();
        if (lineCount == 0) {
            model.addAttribute("error", "파일이 비어 있습니다.");
            return "upload";
        }

        // 정규 표현식 패턴 미리 컴파일
        Pattern datePattern = Pattern.compile("\\b(\\d{4}년 \\d{1,2}월 \\d{1,2}일 화요일)\\b");
        Pattern messagePattern = Pattern.compile("\\[(.*?)\\] \\[(.*?)\\] (.*)");

        // 스레드 풀 최적화
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(availableProcessors * 2);

        AtomicInteger currentLine = new AtomicInteger(0);
        List<CompletableFuture<Void>> futures = new ArrayList<>();

        for (String finalLine : lines) {
            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                try {
                    Matcher dateMatcher = datePattern.matcher(finalLine);
                    if (dateMatcher.find()) {
                        dates.add(dateMatcher.group(1));
                        dateFound[0] = true;
                    }
                    Matcher matcher = messagePattern.matcher(finalLine);
                    if (matcher.matches()) {
                        String user = matcher.group(1); // 날짜와 시간
                        String time = matcher.group(2); // 사용자 이름
                        String message = matcher.group(3); // 메시지 내용

                        userMessageCount.put(user, userMessageCount.getOrDefault(user, 0) + 1);

                        // 비동기적으로 감정 분석 수행
                        String sentiment = sentimentAnalyzer.analyze(message);
                        userSentimentCounts.putIfAbsent(user, new ConcurrentHashMap<>());
                        userSentimentCounts.get(user).merge(sentiment, 1, Integer::sum);
                    }

                    // 진행도 업데이트
                    int progress = (int) ((currentLine.incrementAndGet() / (float) lineCount) * 100);
                    progressWebSocketHandler.sendMessageAsync("Progress: " + progress + "%");

                    // 로깅 추가
                    // System.out.println("Current Line: " + currentLine.get() + " / " + lineCount + " (Progress: " + progress + "%)");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }, executorService);
            futures.add(future);
        }

        // 모든 비동기 작업이 완료될 때까지 대기
        CompletableFuture<Void> allOf = CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]));
        allOf.get();

        executorService.shutdown();
        if (!executorService.awaitTermination(15, TimeUnit.SECONDS)) {
            System.err.println("Executor did not terminate in the specified time.");
            List<Runnable> droppedTasks = executorService.shutdownNow();
            System.err.println("Dropped " + droppedTasks.size() + " tasks");
        }

        List<Map.Entry<String, Integer>> positiveSortedUsers = sortUsersBySentiment(userSentimentCounts, "Positive");
        List<Map.Entry<String, Integer>> negativeSortedUsers = sortUsersBySentiment(userSentimentCounts, "Negative");
        List<Map.Entry<String, Integer>> neutralSortedUsers = sortUsersBySentiment(userSentimentCounts, "Neutral");

        model.addAttribute("sortedUsers", userMessageCount.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                .collect(Collectors.toList()));
        model.addAttribute("date", dates);
        model.addAttribute("positiveSortedUsers", positiveSortedUsers);
        model.addAttribute("negativeSortedUsers", negativeSortedUsers);
        model.addAttribute("neutralSortedUsers", neutralSortedUsers);

        return "result";
    }

    private List<Map.Entry<String, Integer>> sortUsersBySentiment(Map<String, Map<String, Integer>> userSentimentCounts, String sentiment) {
        return userSentimentCounts.entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().getOrDefault(sentiment, 0)))
                .entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                .collect(Collectors.toList());
    }
}
