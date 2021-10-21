package shop.cofin.api.api.user.service;

import shop.cofin.api.api.user.domain.User;
import shop.cofin.api.api.user.domain.UserDTO;

import javax.swing.text.html.Option;
import java.util.Optional;

public interface UserService {
    Optional<User> findById(long userId);
    Optional<String> login (String username, String password);

}
