package com.example.demo;

import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.io.IOException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.LinkedBlockingQueue;

@Component
public class ProgressWebSocketHandler extends TextWebSocketHandler {
    private final CopyOnWriteArrayList<WebSocketSession> sessions = new CopyOnWriteArrayList<>();
    private final BlockingQueue<String> messageQueue = new LinkedBlockingQueue<>();
    private volatile boolean running = true;

    public ProgressWebSocketHandler() {
        new Thread(this::processQueue).start();
    }

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        sessions.add(session);
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
        sessions.remove(session);
    }

    public void sendMessage(String message) {
        messageQueue.add(message);
    }

    public void sendMessageAsync(String message) {
        messageQueue.add(message);
    }

    private void processQueue() {
        while (running) {
            try {
                String message = messageQueue.take();
                for (WebSocketSession session : sessions) {
                    if (session.isOpen()) {
                        CompletableFuture.runAsync(() -> {
                            try {
                                session.sendMessage(new TextMessage(message));
                            } catch (IOException e) {
                                System.err.println("Error sending message: " + e.getMessage());
                                e.printStackTrace();
                                try {
                                    session.close(CloseStatus.SESSION_NOT_RELIABLE);
                                } catch (IOException ioException) {
                                    ioException.printStackTrace();
                                }
                            }
                        });
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    public void stop() {
        running = false;
    }
}
