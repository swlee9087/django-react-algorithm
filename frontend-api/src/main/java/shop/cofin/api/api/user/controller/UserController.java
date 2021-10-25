package shop.cofin.api.api.user.controller;

import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import shop.cofin.api.api.common.controller.CommonController;
import shop.cofin.api.api.user.domain.User;
import shop.cofin.api.api.user.domain.UserDTO;
import shop.cofin.api.api.user.repository.UserRepository;
import shop.cofin.api.api.user.service.UserService;
import org.modelmapper.ModelMapper;

import java.util.List;
import java.util.Optional;

@CrossOrigin("*")
@RequiredArgsConstructor
@RestController
@RequestMapping("/users")
public class UserController implements CommonController<User, Long> {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());
    private final UserService userService;
    private final UserRepository userRepository;

    @PostMapping("/login")
    public ResponseEntity<User> login(@RequestBody User user){
        System.out.println("::: FROM REACT ::: "+user.toString());
        Optional<User> u = userService.login(user.getUsername(),user.getPassword());
        User u2 = u.get();
        System.out.println("::: FROM DB :::"+u2.toString());
//        System.out.println("로그인 :: 리액트에서 넘어온 정보 : " + user.toString());
//        System.out.println("로그인 :: 디비 갔다온애 : " + returnUser.get().toString());
        return ResponseEntity.ok(u2);

    }

    @Override
    public ResponseEntity<List<User>> findAll() {
        return ResponseEntity.ok(userRepository.findAll());
    }

    @Override
    @GetMapping("/{id}")
    public ResponseEntity<User> getById(@PathVariable Long id) {
        return ResponseEntity.ok(userRepository.getById(id));
    }

    @PostMapping
    @Override
    public ResponseEntity<String> save(@RequestBody User user) {
        logger.info(String.format("::: USER INFO ::: %s", user.toString()));
        userRepository.save(user);
        return ResponseEntity.ok("::: USER INFO SAVE SUCCESS :::");
    }

    @PutMapping
    public ResponseEntity<User> update(@RequestBody User user) {
        logger.info(String.format("::: USER INFO EDIT ::: %s", user.toString()));
        userRepository.save(user);
        return ResponseEntity.ok(userRepository.getById(user.getUserId()));
    }

    @Override
    public ResponseEntity<Optional<User>> findById(Long id) {
        System.out.println("::: DETAILS FROM REACT :::");
        return ResponseEntity.ok(userRepository.findById(id));
    }

    @Override
    public ResponseEntity<Boolean> existsById(Long id) {
        System.out.println("::: ID INFO FROM REACT :::");
        return ResponseEntity.ok(userRepository.existsById(id));
    }

    @Override
    public ResponseEntity<Long> count() {
        System.out.println("::: USER COUNT :::");
        return ResponseEntity.ok(userRepository.count());
    }

    @Override
    public ResponseEntity<String> deleteById(Long id) {
        userRepository.deleteById(id);
        return ResponseEntity.ok("::: USER DELETE SUCCESS :::");
    }


}