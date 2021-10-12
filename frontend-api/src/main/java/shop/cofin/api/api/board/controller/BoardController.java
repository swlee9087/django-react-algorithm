package shop.cofin.api.api.board.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.RestController;
import shop.cofin.api.api.board.service.BoardService;

@RestController @RequiredArgsConstructor
public class BoardController {
    private final BoardService boardService;
}
