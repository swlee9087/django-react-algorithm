package shop.cofin.api.api.user.service;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import shop.cofin.api.api.user.domain.User;
import shop.cofin.api.api.user.repository.UserRepository;

import java.util.Optional;

@Service @RequiredArgsConstructor
public class UserServiceImpl implements UserService{

    private final UserRepository userRepository;

    @Override
    public Optional<User> findById(long userId) {
        return userRepository.findById(userId);
    }
}
