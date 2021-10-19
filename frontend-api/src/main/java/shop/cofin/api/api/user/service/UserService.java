package shop.cofin.api.api.user.service;

import shop.cofin.api.api.user.domain.UserSerializer;

import java.util.Optional;

public interface UserService {
    Optional<UserSerializer> findById(long userId);
}
