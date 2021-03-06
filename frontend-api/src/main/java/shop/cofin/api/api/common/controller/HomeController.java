package shop.cofin.api.api.common.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.text.SimpleDateFormat;
import java.util.Date;

@RestController
public final class HomeController {
    @GetMapping("/")
    public String home(){
        return new SimpleDateFormat("yyyy-MM-dd hh:mm:ss").format(new Date());

    }
}