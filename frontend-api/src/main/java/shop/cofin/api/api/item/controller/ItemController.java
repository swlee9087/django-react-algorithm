package shop.cofin.api.api.item.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RestController;
import shop.cofin.api.api.common.controller.CommonController;
import shop.cofin.api.api.item.domain.Item;
import shop.cofin.api.api.item.repository.ItemRepository;
import shop.cofin.api.api.item.service.ItemService;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@RequestMapping("/items")
@RequiredArgsConstructor
@RestController
public class ItemController implements CommonController<Item, Long> {
    private final ItemService itemService;
    private final ItemRepository itemRepository;

    @Override
    public ResponseEntity<List<Item>> findAll() {
        return ResponseEntity.ok(itemRepository.findAll());
    }

    @Override
    public ResponseEntity<Item> getById(Long id) {
        return ResponseEntity.ok(itemRepository.getById(id));
    }

    @Override
    public ResponseEntity<String> save(Item item) {
        itemRepository.save(item);
        return ResponseEntity.ok("::: ITEM SAVE SUCCESS :::");
    }

    @Override
    public ResponseEntity<Optional<Item>> findById(Long id) {
        return ResponseEntity.ok(itemRepository.findById(id));
    }

    @Override
    public ResponseEntity<Boolean> existsById(Long id) {
        return ResponseEntity.ok(itemRepository.existsById(id));
    }

    @Override
    public ResponseEntity<Long> count() {
        return ResponseEntity.ok(itemRepository.count());
    }

    @DeleteMapping("/{id}")
    @Override
    public ResponseEntity<String> deleteById(Long id) {
        itemRepository.deleteById(id);
        return ResponseEntity.ok("::: ID DELETE SUCCESS :::");
    }
}