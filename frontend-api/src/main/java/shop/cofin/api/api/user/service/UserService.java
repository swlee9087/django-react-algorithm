package shop.cofin.api.api.user.service;

import shop.cofin.api.api.user.domain.User;
import java.util.Optional;

public interface UserService {
    Optional<User> findById(long userId);
}
