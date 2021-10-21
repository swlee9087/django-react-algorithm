package shop.cofin.api.api.user.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import shop.cofin.api.api.user.domain.User;
import shop.cofin.api.api.user.domain.UserDTO;
import shop.cofin.api.api.user.service.UserService;

import java.util.Optional;

@CrossOrigin("*")
@RequiredArgsConstructor
@RestController
@RequestMapping("/users")
public class UserController {
    private final UserService userService;

    @PostMapping("/login")
    public ResponseEntity<String> login(@RequestBody UserDTO user){
//        UserDTO returnUser = userService.login(user.getUsername(),user.getPassword());
        Optional<String> returnUser = userService.login(user.getUsername(), user.getPassword());
        System.out.println("Info from MariaDB: "+returnUser.get());
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @GetMapping("/{id}")
    public ResponseEntity<UserDTO> getById(@PathVariable long id)  {
        System.out.println("__________________");
        User user = userService.findById(id).get();
        UserDTO userSerializer = UserDTO.builder()
                .userId(user.getUserId())
                .username(user.getUsername())
                .password(user.getPassword())
                .name(user.getName())
                .email(user.getEmail())
                .regDate(user.getRegDate())
                .build();
        return new ResponseEntity<>(userSerializer, HttpStatus.OK);
    }
}