package shop.cofin.api.api.user.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import shop.cofin.api.api.user.domain.UserSerializer;

import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<UserSerializer, Long> {
    Optional<UserSerializer> findById(long userId);
}
