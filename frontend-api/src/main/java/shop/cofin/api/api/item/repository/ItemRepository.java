package shop.cofin.api.api.item.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import shop.cofin.api.api.item.domain.Item;

@Repository
public interface ItemRepository extends JpaRepository<Item, Long> {
}
